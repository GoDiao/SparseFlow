#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/Exception.h>
#include <immintrin.h>
#include <torch/library.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

using Clock = std::chrono::steady_clock;

enum ProfileCounter : int64_t {
  kRowSumsNs = 0,
  kRowSumsCalls,
  kQuantizeNs,
  kQuantizeCalls,
  kGemvNs,
  kGemvCalls,
  kDynamicNs,
  kDynamicCalls,
  kProfileCounterCount,
};

std::atomic<bool> g_profile_enabled{false};
std::atomic<int64_t> g_profile[kProfileCounterCount];

inline int64_t elapsed_ns(Clock::time_point started) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - started).count();
}

inline void add_profile(ProfileCounter elapsed, ProfileCounter calls, Clock::time_point started) {
  if (!g_profile_enabled.load(std::memory_order_relaxed)) {
    return;
  }
  g_profile[elapsed].fetch_add(elapsed_ns(started), std::memory_order_relaxed);
  g_profile[calls].fetch_add(1, std::memory_order_relaxed);
}

at::Tensor row_sums_cpu(const at::Tensor& weight) {
  const auto profile_started = Clock::now();
  TORCH_CHECK(weight.device().is_cpu(), "weight must be on CPU");
  TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8");
  TORCH_CHECK(weight.dim() == 2 && weight.is_contiguous(), "weight must be contiguous [N,K]");
  const auto n = weight.size(0);
  const auto k = weight.size(1);
  auto result = at::empty({n}, weight.options().dtype(at::kInt));
  const auto* data = weight.const_data_ptr<int8_t>();
  auto* output = result.mutable_data_ptr<int32_t>();
  at::parallel_for(0, n, 16, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const int8_t* values = data + row * k;
      int32_t sum = 0;
      int64_t column = 0;
      for (; column + 64 <= k; column += 64) {
        const __m512i packed = _mm512_loadu_si512(values + column);
        const __m256i low = _mm512_castsi512_si256(packed);
        const __m256i high = _mm512_extracti64x4_epi64(packed, 1);
        const __m512i low16 = _mm512_cvtepi8_epi16(low);
        const __m512i high16 = _mm512_cvtepi8_epi16(high);
        sum += _mm512_reduce_add_epi32(_mm512_madd_epi16(low16, _mm512_set1_epi16(1)));
        sum += _mm512_reduce_add_epi32(_mm512_madd_epi16(high16, _mm512_set1_epi16(1)));
      }
      for (; column < k; ++column) {
        sum += values[column];
      }
      output[row] = sum;
    }
  });
  add_profile(kRowSumsNs, kRowSumsCalls, profile_started);
  return result;
}

at::Tensor dynamic_linear_cpu(
    const at::Tensor& input,
    const at::Tensor& weight,
    const at::Tensor& weight_scales,
    const at::Tensor& weight_row_sums) {
  const auto dynamic_started = Clock::now();
  TORCH_CHECK(input.device().is_cpu() && weight.device().is_cpu(), "tensors must be on CPU");
  TORCH_CHECK(input.scalar_type() == at::kFloat, "input must be float32");
  TORCH_CHECK(weight.scalar_type() == at::kChar, "weight must be int8");
  TORCH_CHECK(weight_scales.scalar_type() == at::kFloat, "weight_scales must be float32");
  TORCH_CHECK(weight_row_sums.scalar_type() == at::kInt, "weight_row_sums must be int32");
  TORCH_CHECK(input.dim() == 2 && weight.dim() == 2, "input and weight must be matrices");
  TORCH_CHECK(input.is_contiguous() && weight.is_contiguous(), "input and weight must be contiguous");
  TORCH_CHECK(weight_scales.is_contiguous() && weight_row_sums.is_contiguous(), "metadata must be contiguous");
  TORCH_CHECK(input.size(1) == weight.size(1), "input/weight K mismatch");
  TORCH_CHECK(weight_scales.numel() == weight.size(0), "scale/output mismatch");
  TORCH_CHECK(weight_row_sums.numel() == weight.size(0), "row-sum/output mismatch");

  const int64_t m = input.size(0);
  const int64_t n = weight.size(0);
  const int64_t k = weight.size(1);
  auto quantized = at::empty({m, k}, input.options().dtype(at::kByte));
  auto input_scales = at::empty({m}, input.options());
  auto zero_points = at::empty({m}, input.options().dtype(at::kInt));
  const float* input_data = input.const_data_ptr<float>();
  auto* quantized_data = quantized.mutable_data_ptr<uint8_t>();
  auto* input_scale_data = input_scales.mutable_data_ptr<float>();
  auto* zero_point_data = zero_points.mutable_data_ptr<int32_t>();

  const auto quantize_started = Clock::now();
  at::parallel_for(0, m, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const float* source = input_data + row * k;
      uint8_t* target = quantized_data + row * k;
      float minimum = 0.0f;
      float maximum = 0.0f;
      for (int64_t column = 0; column < k; ++column) {
        minimum = std::min(minimum, source[column]);
        maximum = std::max(maximum, source[column]);
      }
      const float scale = maximum > minimum ? (maximum - minimum) / 255.0f : 1.0f;
      const int32_t zero_point = std::clamp<int32_t>(
          static_cast<int32_t>(std::nearbyint(-minimum / scale)), 0, 255);
      input_scale_data[row] = scale;
      zero_point_data[row] = zero_point;
      for (int64_t column = 0; column < k; ++column) {
        const int32_t value = std::clamp<int32_t>(
            static_cast<int32_t>(std::nearbyint(source[column] / scale)) + zero_point,
            0,
            255);
        target[column] = static_cast<uint8_t>(value);
      }
    }
  });
  add_profile(kQuantizeNs, kQuantizeCalls, quantize_started);

  const auto gemv_started = Clock::now();
  auto output = at::empty({m, n}, input.options());
  const auto* weight_data = weight.const_data_ptr<int8_t>();
  const auto* weight_scale_data = weight_scales.const_data_ptr<float>();
  const auto* row_sum_data = weight_row_sums.const_data_ptr<int32_t>();
  auto* output_data = output.mutable_data_ptr<float>();
  at::parallel_for(0, m * n, 8, [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t row = index / n;
      const int64_t output_channel = index % n;
      const uint8_t* activation = quantized_data + row * k;
      const int8_t* weights = weight_data + output_channel * k;
      __m512i accumulator = _mm512_setzero_si512();
      int64_t column = 0;
      for (; column + 64 <= k; column += 64) {
        const __m512i a = _mm512_loadu_si512(activation + column);
        const __m512i w = _mm512_loadu_si512(weights + column);
        accumulator = _mm512_dpbusd_epi32(accumulator, a, w);
      }
      int32_t dot = _mm512_reduce_add_epi32(accumulator);
      for (; column < k; ++column) {
        dot += static_cast<int32_t>(activation[column]) * static_cast<int32_t>(weights[column]);
      }
      dot -= zero_point_data[row] * row_sum_data[output_channel];
      output_data[index] = static_cast<float>(dot) * input_scale_data[row] *
          weight_scale_data[output_channel];
    }
  });
  add_profile(kGemvNs, kGemvCalls, gemv_started);
  add_profile(kDynamicNs, kDynamicCalls, dynamic_started);
  return output;
}

void set_profile_enabled(const at::Tensor& anchor, bool enabled) {
  TORCH_CHECK(anchor.device().is_cpu(), "profile anchor must be on CPU");
  g_profile_enabled.store(enabled, std::memory_order_relaxed);
}

void reset_profile(const at::Tensor& anchor) {
  TORCH_CHECK(anchor.device().is_cpu(), "profile anchor must be on CPU");
  for (auto& counter : g_profile) {
    counter.store(0, std::memory_order_relaxed);
  }
}

at::Tensor profile_snapshot(const at::Tensor& anchor) {
  TORCH_CHECK(anchor.device().is_cpu(), "profile anchor must be on CPU");
  auto result = at::empty({kProfileCounterCount}, at::TensorOptions().dtype(at::kLong));
  auto* output = result.mutable_data_ptr<int64_t>();
  for (int64_t index = 0; index < kProfileCounterCount; ++index) {
    output[index] = g_profile[index].load(std::memory_order_relaxed);
  }
  return result;
}

}  // namespace

TORCH_LIBRARY(sparseflow_native, m) {
  m.def("row_sums(Tensor weight) -> Tensor");
  m.def("dynamic_linear(Tensor input, Tensor weight, Tensor weight_scales, Tensor weight_row_sums) -> Tensor");
  m.def("set_profile_enabled(Tensor anchor, bool enabled) -> ()");
  m.def("reset_profile(Tensor anchor) -> ()");
  m.def("profile_snapshot(Tensor anchor) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparseflow_native, CPU, m) {
  m.impl("row_sums", &row_sums_cpu);
  m.impl("dynamic_linear", &dynamic_linear_cpu);
  m.impl("set_profile_enabled", &set_profile_enabled);
  m.impl("reset_profile", &reset_profile);
  m.impl("profile_snapshot", &profile_snapshot);
}

// [Main Dev]
