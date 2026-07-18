#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/util/BFloat16.h>
#include <immintrin.h>
#include <torch/library.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

struct QuantizedRows {
  at::Tensor data;
  at::Tensor scales;
  at::Tensor zero_points;
};

QuantizedRows quantize_rows(const at::Tensor& input) {
  TORCH_CHECK(input.scalar_type() == at::kFloat && input.dim() == 2, "input must be F32 [M,K]");
  TORCH_CHECK(input.device().is_cpu() && input.is_contiguous(), "input must be contiguous CPU");
  const int64_t rows = input.size(0);
  const int64_t columns = input.size(1);
  auto data = at::empty({rows, columns}, input.options().dtype(at::kByte));
  auto scales = at::empty({rows}, input.options());
  auto zero_points = at::empty({rows}, input.options().dtype(at::kInt));
  const float* source = input.const_data_ptr<float>();
  auto* target = data.mutable_data_ptr<uint8_t>();
  auto* scale_data = scales.mutable_data_ptr<float>();
  auto* zero_data = zero_points.mutable_data_ptr<int32_t>();
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const float* values = source + row * columns;
      uint8_t* output = target + row * columns;
      float minimum = 0.0f;
      float maximum = 0.0f;
      for (int64_t column = 0; column < columns; ++column) {
        minimum = std::min(minimum, values[column]);
        maximum = std::max(maximum, values[column]);
      }
      const float scale = maximum > minimum ? (maximum - minimum) / 255.0f : 1.0f;
      const int32_t zero_point = std::clamp<int32_t>(
          static_cast<int32_t>(std::nearbyint(-minimum / scale)), 0, 255);
      scale_data[row] = scale;
      zero_data[row] = zero_point;
      for (int64_t column = 0; column < columns; ++column) {
        output[column] = static_cast<uint8_t>(std::clamp<int32_t>(
            static_cast<int32_t>(std::nearbyint(values[column] / scale)) + zero_point,
            0,
            255));
      }
    }
  });
  return {data, scales, zero_points};
}

inline int32_t dot_u8_s8(const uint8_t* activation, const int8_t* weight, int64_t columns) {
  __m512i accumulator = _mm512_setzero_si512();
  int64_t column = 0;
  for (; column + 64 <= columns; column += 64) {
    const __m512i a = _mm512_loadu_si512(activation + column);
    const __m512i w = _mm512_loadu_si512(weight + column);
    accumulator = _mm512_dpbusd_epi32(accumulator, a, w);
  }
  int32_t result = _mm512_reduce_add_epi32(accumulator);
  for (; column < columns; ++column) {
    result += static_cast<int32_t>(activation[column]) * static_cast<int32_t>(weight[column]);
  }
  return result;
}

at::Tensor fused_moe_cpu(
    const at::Tensor& hidden_states,
    const at::Tensor& selected_experts,
    const at::Tensor& routing_weights,
    const at::Tensor& expert_ids,
    at::TensorList gate_up_weights,
    at::TensorList gate_up_scales,
    at::TensorList gate_up_row_sums,
    at::TensorList down_weights,
    at::TensorList down_scales,
    at::TensorList down_row_sums) {
  TORCH_CHECK(hidden_states.device().is_cpu(), "hidden states must be on CPU");
  TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16, "hidden states must be BF16");
  TORCH_CHECK(hidden_states.dim() == 2 && hidden_states.is_contiguous(), "hidden states must be contiguous [M,K]");
  TORCH_CHECK(selected_experts.scalar_type() == at::kLong && selected_experts.dim() == 2, "selected experts must be I64 [M,top_k]");
  TORCH_CHECK(selected_experts.is_contiguous(), "selected experts must be contiguous");
  TORCH_CHECK(routing_weights.sizes() == selected_experts.sizes(), "routing shape mismatch");
  TORCH_CHECK(expert_ids.scalar_type() == at::kLong && expert_ids.dim() == 1, "expert IDs must be I64");
  const int64_t unique = expert_ids.numel();
  TORCH_CHECK(unique > 0, "fused MoE requires at least one expert");
  TORCH_CHECK(
      gate_up_weights.size() == unique && gate_up_scales.size() == unique &&
          gate_up_row_sums.size() == unique && down_weights.size() == unique &&
          down_scales.size() == unique && down_row_sums.size() == unique,
      "expert tensor-list length mismatch");

  const int64_t rows = hidden_states.size(0);
  const int64_t hidden = hidden_states.size(1);
  const int64_t top_k = selected_experts.size(1);
  const int64_t assignments = rows * top_k;
  const int64_t gate_rows = gate_up_weights[0].size(0);
  TORCH_CHECK(gate_rows % 2 == 0, "gate/up output must be even");
  const int64_t intermediate = gate_rows / 2;

  std::vector<const int8_t*> gate_weights(unique);
  std::vector<const float*> gate_scales(unique);
  std::vector<const int32_t*> gate_sums(unique);
  std::vector<const int8_t*> down_weight_data(unique);
  std::vector<const float*> down_scale_data(unique);
  std::vector<const int32_t*> down_sum_data(unique);
  std::vector<int64_t> ids(unique);
  const int64_t* id_data = expert_ids.contiguous().const_data_ptr<int64_t>();
  for (int64_t slot = 0; slot < unique; ++slot) {
    const auto& gate_weight = gate_up_weights[slot];
    const auto& gate_scale = gate_up_scales[slot];
    const auto& gate_sum = gate_up_row_sums[slot];
    const auto& down_weight = down_weights[slot];
    const auto& down_scale = down_scales[slot];
    const auto& down_sum = down_row_sums[slot];
    TORCH_CHECK(gate_weight.scalar_type() == at::kChar && gate_weight.is_contiguous(), "gate weight must be contiguous I8");
    TORCH_CHECK(gate_weight.sizes() == at::IntArrayRef({gate_rows, hidden}), "gate weight shape mismatch");
    TORCH_CHECK(gate_scale.scalar_type() == at::kFloat && gate_scale.numel() == gate_rows, "gate scale mismatch");
    TORCH_CHECK(gate_sum.scalar_type() == at::kInt && gate_sum.numel() == gate_rows, "gate row-sum mismatch");
    TORCH_CHECK(down_weight.scalar_type() == at::kChar && down_weight.is_contiguous(), "down weight must be contiguous I8");
    TORCH_CHECK(down_weight.sizes() == at::IntArrayRef({hidden, intermediate}), "down weight shape mismatch");
    TORCH_CHECK(down_scale.scalar_type() == at::kFloat && down_scale.numel() == hidden, "down scale mismatch");
    TORCH_CHECK(down_sum.scalar_type() == at::kInt && down_sum.numel() == hidden, "down row-sum mismatch");
    ids[slot] = id_data[slot];
    gate_weights[slot] = gate_weight.const_data_ptr<int8_t>();
    gate_scales[slot] = gate_scale.const_data_ptr<float>();
    gate_sums[slot] = gate_sum.const_data_ptr<int32_t>();
    down_weight_data[slot] = down_weight.const_data_ptr<int8_t>();
    down_scale_data[slot] = down_scale.const_data_ptr<float>();
    down_sum_data[slot] = down_sum.const_data_ptr<int32_t>();
  }

  const auto selected = selected_experts.contiguous();
  const int64_t* selected_data = selected.const_data_ptr<int64_t>();
  std::vector<int64_t> assignment_slots(assignments, -1);
  for (int64_t assignment = 0; assignment < assignments; ++assignment) {
    const int64_t expert = selected_data[assignment];
    auto iterator = std::lower_bound(ids.begin(), ids.end(), expert);
    TORCH_CHECK(iterator != ids.end() && *iterator == expert, "selected expert missing from native batch");
    assignment_slots[assignment] = std::distance(ids.begin(), iterator);
  }
  std::vector<int64_t> compute_order(assignments);
  std::iota(compute_order.begin(), compute_order.end(), 0);
  std::stable_sort(compute_order.begin(), compute_order.end(), [&](int64_t left, int64_t right) {
    return assignment_slots[left] < assignment_slots[right];
  });

  auto hidden_float = hidden_states.to(at::kFloat).contiguous();
  auto hidden_quantized = quantize_rows(hidden_float);
  const auto* hidden_q = hidden_quantized.data.const_data_ptr<uint8_t>();
  const auto* hidden_scales = hidden_quantized.scales.const_data_ptr<float>();
  const auto* hidden_zero = hidden_quantized.zero_points.const_data_ptr<int32_t>();
  auto projected = at::empty({assignments, gate_rows}, hidden_float.options());
  float* projected_data = projected.mutable_data_ptr<float>();
  at::parallel_for(0, assignments * intermediate, 8, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t ordered = task / intermediate;
      const int64_t channel = task % intermediate;
      const int64_t assignment = compute_order[ordered];
      const int64_t token = assignment / top_k;
      const int64_t slot = assignment_slots[assignment];
      const uint8_t* input = hidden_q + token * hidden;
      const int8_t* gate = gate_weights[slot] + channel * hidden;
      const int8_t* up = gate_weights[slot] + (channel + intermediate) * hidden;
      const float input_scale = hidden_scales[token];
      const int32_t zero = hidden_zero[token];
      const float gate_value =
          static_cast<float>(dot_u8_s8(input, gate, hidden) - zero * gate_sums[slot][channel]) *
          input_scale * gate_scales[slot][channel];
      const float up_value =
          static_cast<float>(dot_u8_s8(input, up, hidden) - zero * gate_sums[slot][channel + intermediate]) *
          input_scale * gate_scales[slot][channel + intermediate];
      projected_data[assignment * gate_rows + channel] = gate_value;
      projected_data[assignment * gate_rows + channel + intermediate] = up_value;
    }
  });

  auto activated = at::silu(projected.narrow(1, 0, intermediate)) *
      projected.narrow(1, intermediate, intermediate);

  auto activated_quantized = quantize_rows(activated);
  const auto* activated_q = activated_quantized.data.const_data_ptr<uint8_t>();
  const auto* activated_scales = activated_quantized.scales.const_data_ptr<float>();
  const auto* activated_zero = activated_quantized.zero_points.const_data_ptr<int32_t>();
  auto routing = routing_weights.to(at::kFloat).contiguous();
  const float* routing_data = routing.const_data_ptr<float>();
  std::vector<int64_t> token_order(assignments);
  for (int64_t token = 0; token < rows; ++token) {
    for (int64_t position = 0; position < top_k; ++position) {
      token_order[token * top_k + position] = token * top_k + position;
    }
    std::stable_sort(
        token_order.begin() + token * top_k,
        token_order.begin() + (token + 1) * top_k,
        [&](int64_t left, int64_t right) {
          return assignment_slots[left] < assignment_slots[right];
        });
  }

  if (rows == 1) {
    auto output = at::zeros({1, hidden}, hidden_states.options());
    auto* output_data = output.mutable_data_ptr<c10::BFloat16>();
    at::parallel_for(0, hidden, 8, [&](int64_t begin, int64_t end) {
      for (int64_t channel = begin; channel < end; ++channel) {
        c10::BFloat16 accumulated = 0.0f;
        for (int64_t order = 0; order < top_k; ++order) {
          const int64_t assignment = token_order[order];
          const int64_t slot = assignment_slots[assignment];
          const uint8_t* input = activated_q + assignment * intermediate;
          const int8_t* weight = down_weight_data[slot] + channel * intermediate;
          const int32_t dot = dot_u8_s8(input, weight, intermediate) -
              activated_zero[assignment] * down_sum_data[slot][channel];
          const float value = static_cast<float>(dot) * activated_scales[assignment] *
              down_scale_data[slot][channel];
          const c10::BFloat16 expert_value = value;
          const c10::BFloat16 route_value = routing_data[assignment];
          const c10::BFloat16 weighted =
              static_cast<float>(expert_value) * static_cast<float>(route_value);
          accumulated = static_cast<float>(accumulated) + static_cast<float>(weighted);
        }
        output_data[channel] = accumulated;
      }
    });
    return output;
  }

  auto contributions = at::empty({assignments, hidden}, hidden_float.options());
  float* contribution_data = contributions.mutable_data_ptr<float>();
  at::parallel_for(0, assignments * hidden, 8, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t ordered = task / hidden;
      const int64_t channel = task % hidden;
      const int64_t assignment = compute_order[ordered];
      const int64_t slot = assignment_slots[assignment];
      const uint8_t* input = activated_q + assignment * intermediate;
      const int8_t* weight = down_weight_data[slot] + channel * intermediate;
      const int32_t dot = dot_u8_s8(input, weight, intermediate) -
          activated_zero[assignment] * down_sum_data[slot][channel];
      contribution_data[assignment * hidden + channel] =
          static_cast<float>(dot) * activated_scales[assignment] * down_scale_data[slot][channel];
    }
  });

  auto output = at::zeros({rows, hidden}, hidden_states.options());
  auto* output_data = output.mutable_data_ptr<c10::BFloat16>();
  at::parallel_for(0, rows * hidden, 64, [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t token = index / hidden;
      const int64_t channel = index % hidden;
      c10::BFloat16 accumulated = 0.0f;
      for (int64_t order = 0; order < top_k; ++order) {
        const int64_t assignment = token_order[token * top_k + order];
        const c10::BFloat16 expert_value = contribution_data[assignment * hidden + channel];
        const c10::BFloat16 route_value = routing_data[assignment];
        const c10::BFloat16 weighted = static_cast<float>(expert_value) * static_cast<float>(route_value);
        accumulated = static_cast<float>(accumulated) + static_cast<float>(weighted);
      }
      output_data[index] = accumulated;
    }
  });
  return output;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sparseflow_native, m) {
  m.def(
      "fused_moe(Tensor hidden_states, Tensor selected_experts, Tensor routing_weights, "
      "Tensor expert_ids, Tensor[] gate_up_weights, Tensor[] gate_up_scales, "
      "Tensor[] gate_up_row_sums, Tensor[] down_weights, Tensor[] down_scales, "
      "Tensor[] down_row_sums) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparseflow_native, CPU, m) {
  m.impl("fused_moe", &fused_moe_cpu);
}

// [Main Dev]
