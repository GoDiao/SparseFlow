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

// [Main Dev]
// This operator keeps the Stage 7.6 W8A8 arithmetic but changes the work
// layout: one task owns an expert/output-channel pair and processes every row
// routed to that expert before moving to the next channel.

namespace {

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

void check_workspace_tensor(
    const at::Tensor& tensor,
    at::ScalarType dtype,
    const char* name) {
  TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous(), name, " must be contiguous CPU");
  TORCH_CHECK(tensor.scalar_type() == dtype, name, " has unexpected dtype");
}

void quantize_float_rows(
    const float* source,
    int64_t rows,
    int64_t columns,
    uint8_t* target,
    float* scales,
    int32_t* zero_points) {
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
      scales[row] = scale;
      zero_points[row] = zero_point;
      for (int64_t column = 0; column < columns; ++column) {
        output[column] = static_cast<uint8_t>(std::clamp<int32_t>(
            static_cast<int32_t>(std::nearbyint(values[column] / scale)) + zero_point,
            0,
            255));
      }
    }
  });
}

void quantize_bf16_rows(
    const at::Tensor& input,
    uint8_t* target,
    float* scales,
    int32_t* zero_points) {
  const int64_t rows = input.size(0);
  const int64_t columns = input.size(1);
  const auto* source = input.const_data_ptr<c10::BFloat16>();
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t row = begin; row < end; ++row) {
      const auto* values = source + row * columns;
      uint8_t* output = target + row * columns;
      float minimum = 0.0f;
      float maximum = 0.0f;
      for (int64_t column = 0; column < columns; ++column) {
        const float value = static_cast<float>(values[column]);
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
      }
      const float scale = maximum > minimum ? (maximum - minimum) / 255.0f : 1.0f;
      const int32_t zero_point = std::clamp<int32_t>(
          static_cast<int32_t>(std::nearbyint(-minimum / scale)), 0, 255);
      scales[row] = scale;
      zero_points[row] = zero_point;
      for (int64_t column = 0; column < columns; ++column) {
        output[column] = static_cast<uint8_t>(std::clamp<int32_t>(
            static_cast<int32_t>(std::nearbyint(static_cast<float>(values[column]) / scale)) + zero_point,
            0,
            255));
      }
    }
  });
}

at::Tensor grouped_moe_cpu(
    const at::Tensor& hidden_states,
    const at::Tensor& selected_experts,
    const at::Tensor& routing_weights,
    const at::Tensor& expert_ids,
    at::TensorList gate_up_weights,
    at::TensorList gate_up_scales,
    at::TensorList gate_up_row_sums,
    at::TensorList down_weights,
    at::TensorList down_scales,
    at::TensorList down_row_sums,
    const at::Tensor& plan_counts,
    const at::Tensor& plan_offsets,
    const at::Tensor& plan_assignments,
    const at::Tensor& plan_rows,
    const at::Tensor& plan_slots,
    const at::Tensor& plan_token_order,
    const at::Tensor& hidden_quantized,
    const at::Tensor& hidden_scales,
    const at::Tensor& hidden_zero_points,
    const at::Tensor& projected,
    const at::Tensor& activated,
    const at::Tensor& activated_quantized,
    const at::Tensor& activated_scales,
    const at::Tensor& activated_zero_points,
    const at::Tensor& contributions,
    const at::Tensor& output) {
  TORCH_CHECK(hidden_states.device().is_cpu() && hidden_states.is_contiguous(), "hidden states must be contiguous CPU");
  TORCH_CHECK(hidden_states.scalar_type() == at::kBFloat16 && hidden_states.dim() == 2, "hidden states must be BF16 [M,K]");
  TORCH_CHECK(selected_experts.scalar_type() == at::kLong && selected_experts.dim() == 2 && selected_experts.is_contiguous(), "selected experts must be I64 [M,top_k]");
  TORCH_CHECK(routing_weights.sizes() == selected_experts.sizes(), "routing shape mismatch");
  TORCH_CHECK(expert_ids.scalar_type() == at::kLong && expert_ids.dim() == 1 && expert_ids.is_contiguous(), "expert IDs must be contiguous I64");

  const int64_t rows = hidden_states.size(0);
  const int64_t hidden = hidden_states.size(1);
  const int64_t top_k = selected_experts.size(1);
  const int64_t assignments = rows * top_k;
  const int64_t unique = expert_ids.numel();
  TORCH_CHECK(rows > 0 && top_k > 0 && unique > 0, "grouped MoE requires non-empty inputs");
  TORCH_CHECK(
      gate_up_weights.size() == unique && gate_up_scales.size() == unique &&
          gate_up_row_sums.size() == unique && down_weights.size() == unique &&
          down_scales.size() == unique && down_row_sums.size() == unique,
      "expert tensor-list length mismatch");

  const int64_t gate_rows = gate_up_weights[0].size(0);
  TORCH_CHECK(gate_rows % 2 == 0, "gate/up output must be even");
  const int64_t intermediate = gate_rows / 2;
  check_workspace_tensor(plan_counts, at::kLong, "plan_counts");
  check_workspace_tensor(plan_offsets, at::kLong, "plan_offsets");
  check_workspace_tensor(plan_assignments, at::kLong, "plan_assignments");
  check_workspace_tensor(plan_rows, at::kLong, "plan_rows");
  check_workspace_tensor(plan_slots, at::kLong, "plan_slots");
  check_workspace_tensor(plan_token_order, at::kLong, "plan_token_order");
  check_workspace_tensor(hidden_quantized, at::kByte, "hidden_quantized");
  check_workspace_tensor(hidden_scales, at::kFloat, "hidden_scales");
  check_workspace_tensor(hidden_zero_points, at::kInt, "hidden_zero_points");
  check_workspace_tensor(projected, at::kFloat, "projected");
  check_workspace_tensor(activated, at::kFloat, "activated");
  check_workspace_tensor(activated_quantized, at::kByte, "activated_quantized");
  check_workspace_tensor(activated_scales, at::kFloat, "activated_scales");
  check_workspace_tensor(activated_zero_points, at::kInt, "activated_zero_points");
  check_workspace_tensor(contributions, at::kFloat, "contributions");
  check_workspace_tensor(output, at::kBFloat16, "output");
  TORCH_CHECK(plan_counts.numel() >= unique && plan_offsets.numel() >= unique + 1, "group plan expert capacity is too small");
  TORCH_CHECK(
      plan_assignments.numel() >= assignments && plan_rows.numel() >= assignments &&
          plan_slots.numel() >= assignments && plan_token_order.numel() >= assignments,
      "group plan assignment capacity is too small");
  TORCH_CHECK(hidden_quantized.dim() == 2 && hidden_quantized.size(0) >= rows && hidden_quantized.size(1) >= hidden, "hidden quantization workspace is too small");
  TORCH_CHECK(hidden_scales.numel() >= rows && hidden_zero_points.numel() >= rows, "hidden metadata workspace is too small");
  TORCH_CHECK(projected.dim() == 2 && projected.size(0) >= assignments && projected.size(1) >= gate_rows, "projection workspace is too small");
  TORCH_CHECK(activated.dim() == 2 && activated.size(0) >= assignments && activated.size(1) >= intermediate, "activation workspace is too small");
  TORCH_CHECK(activated_quantized.dim() == 2 && activated_quantized.size(0) >= assignments && activated_quantized.size(1) >= intermediate, "activation quantization workspace is too small");
  TORCH_CHECK(activated_scales.numel() >= assignments && activated_zero_points.numel() >= assignments, "activation metadata workspace is too small");
  TORCH_CHECK(contributions.dim() == 2 && contributions.size(0) >= assignments && contributions.size(1) >= hidden, "contribution workspace is too small");
  TORCH_CHECK(output.dim() == 2 && output.size(0) >= rows && output.size(1) >= hidden, "output workspace is too small");

  std::vector<const int8_t*> gate_weights(unique);
  std::vector<const float*> gate_scales(unique);
  std::vector<const int32_t*> gate_sums(unique);
  std::vector<const int8_t*> down_weight_data(unique);
  std::vector<const float*> down_scale_data(unique);
  std::vector<const int32_t*> down_sum_data(unique);
  std::vector<int64_t> ids(unique);
  const int64_t* id_data = expert_ids.const_data_ptr<int64_t>();
  for (int64_t slot = 0; slot < unique; ++slot) {
    const auto& gate_weight = gate_up_weights[slot];
    const auto& gate_scale = gate_up_scales[slot];
    const auto& gate_sum = gate_up_row_sums[slot];
    const auto& down_weight = down_weights[slot];
    const auto& down_scale = down_scales[slot];
    const auto& down_sum = down_row_sums[slot];
    TORCH_CHECK(gate_weight.scalar_type() == at::kChar && gate_weight.is_contiguous() && gate_weight.sizes() == at::IntArrayRef({gate_rows, hidden}), "gate weight mismatch");
    TORCH_CHECK(gate_scale.scalar_type() == at::kFloat && gate_scale.numel() == gate_rows, "gate scale mismatch");
    TORCH_CHECK(gate_sum.scalar_type() == at::kInt && gate_sum.numel() == gate_rows, "gate row sum mismatch");
    TORCH_CHECK(down_weight.scalar_type() == at::kChar && down_weight.is_contiguous() && down_weight.sizes() == at::IntArrayRef({hidden, intermediate}), "down weight mismatch");
    TORCH_CHECK(down_scale.scalar_type() == at::kFloat && down_scale.numel() == hidden, "down scale mismatch");
    TORCH_CHECK(down_sum.scalar_type() == at::kInt && down_sum.numel() == hidden, "down row sum mismatch");
    if (slot > 0) {
      TORCH_CHECK(ids[slot - 1] < id_data[slot], "expert IDs must be strictly increasing");
    }
    ids[slot] = id_data[slot];
    gate_weights[slot] = gate_weight.const_data_ptr<int8_t>();
    gate_scales[slot] = gate_scale.const_data_ptr<float>();
    gate_sums[slot] = gate_sum.const_data_ptr<int32_t>();
    down_weight_data[slot] = down_weight.const_data_ptr<int8_t>();
    down_scale_data[slot] = down_scale.const_data_ptr<float>();
    down_sum_data[slot] = down_sum.const_data_ptr<int32_t>();
  }

  const int64_t* selected_data = selected_experts.const_data_ptr<int64_t>();
  auto* counts = plan_counts.mutable_data_ptr<int64_t>();
  auto* offsets = plan_offsets.mutable_data_ptr<int64_t>();
  auto* group_assignments = plan_assignments.mutable_data_ptr<int64_t>();
  auto* group_rows = plan_rows.mutable_data_ptr<int64_t>();
  auto* group_slots = plan_slots.mutable_data_ptr<int64_t>();
  auto* token_order = plan_token_order.mutable_data_ptr<int64_t>();
  std::fill(counts, counts + unique, 0);
  std::vector<int64_t> assignment_slots(assignments, -1);
  for (int64_t assignment = 0; assignment < assignments; ++assignment) {
    const auto iterator = std::lower_bound(ids.begin(), ids.end(), selected_data[assignment]);
    TORCH_CHECK(iterator != ids.end() && *iterator == selected_data[assignment], "selected expert missing from native batch");
    const int64_t slot = std::distance(ids.begin(), iterator);
    assignment_slots[assignment] = slot;
    counts[slot] += 1;
  }
  offsets[0] = 0;
  for (int64_t slot = 0; slot < unique; ++slot) {
    offsets[slot + 1] = offsets[slot] + counts[slot];
  }
  std::vector<int64_t> cursors(offsets, offsets + unique);
  for (int64_t assignment = 0; assignment < assignments; ++assignment) {
    const int64_t slot = assignment_slots[assignment];
    const int64_t position = cursors[slot]++;
    group_assignments[position] = assignment;
    group_rows[position] = assignment / top_k;
    group_slots[position] = slot;
  }
  for (int64_t token = 0; token < rows; ++token) {
    for (int64_t position = 0; position < top_k; ++position) {
      token_order[token * top_k + position] = token * top_k + position;
    }
    std::stable_sort(
        token_order + token * top_k,
        token_order + (token + 1) * top_k,
        [&](int64_t left, int64_t right) { return assignment_slots[left] < assignment_slots[right]; });
  }

  auto hidden_q = hidden_quantized.narrow(0, 0, rows).narrow(1, 0, hidden);
  auto hidden_scale_view = hidden_scales.narrow(0, 0, rows);
  auto hidden_zero_view = hidden_zero_points.narrow(0, 0, rows);
  quantize_bf16_rows(
      hidden_states,
      hidden_q.mutable_data_ptr<uint8_t>(),
      hidden_scale_view.mutable_data_ptr<float>(),
      hidden_zero_view.mutable_data_ptr<int32_t>());

  auto projected_view = projected.narrow(0, 0, assignments).narrow(1, 0, gate_rows);
  float* projected_data = projected_view.mutable_data_ptr<float>();
  const uint8_t* hidden_q_data = hidden_q.const_data_ptr<uint8_t>();
  const float* hidden_scale_data = hidden_scale_view.const_data_ptr<float>();
  const int32_t* hidden_zero_data = hidden_zero_view.const_data_ptr<int32_t>();
  at::parallel_for(0, unique * intermediate, 1, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t slot = task / intermediate;
      const int64_t channel = task % intermediate;
      const int64_t first = offsets[slot];
      const int64_t last = offsets[slot + 1];
      const int8_t* gate = gate_weights[slot] + channel * hidden;
      const int8_t* up = gate_weights[slot] + (channel + intermediate) * hidden;
      for (int64_t position = first; position < last; ++position) {
        const int64_t assignment = group_assignments[position];
        const int64_t row = group_rows[position];
        const uint8_t* input = hidden_q_data + row * hidden;
        const float input_scale = hidden_scale_data[row];
        const int32_t zero = hidden_zero_data[row];
        projected_data[assignment * gate_rows + channel] =
            static_cast<float>(dot_u8_s8(input, gate, hidden) - zero * gate_sums[slot][channel]) *
            input_scale * gate_scales[slot][channel];
        projected_data[assignment * gate_rows + channel + intermediate] =
            static_cast<float>(dot_u8_s8(input, up, hidden) - zero * gate_sums[slot][channel + intermediate]) *
            input_scale * gate_scales[slot][channel + intermediate];
      }
    }
  });

  auto activated_view = activated.narrow(0, 0, assignments).narrow(1, 0, intermediate);
  auto gates = projected_view.narrow(1, 0, intermediate);
  auto ups = projected_view.narrow(1, intermediate, intermediate);
  activated_view.copy_(at::silu(gates));
  activated_view.mul_(ups);

  auto activated_q = activated_quantized.narrow(0, 0, assignments).narrow(1, 0, intermediate);
  auto activated_scale_view = activated_scales.narrow(0, 0, assignments);
  auto activated_zero_view = activated_zero_points.narrow(0, 0, assignments);
  quantize_float_rows(
      activated_view.const_data_ptr<float>(),
      assignments,
      intermediate,
      activated_q.mutable_data_ptr<uint8_t>(),
      activated_scale_view.mutable_data_ptr<float>(),
      activated_zero_view.mutable_data_ptr<int32_t>());

  auto contributions_view = contributions.narrow(0, 0, assignments).narrow(1, 0, hidden);
  float* contribution_data = contributions_view.mutable_data_ptr<float>();
  const uint8_t* activated_q_data = activated_q.const_data_ptr<uint8_t>();
  const float* activated_scale_data = activated_scale_view.const_data_ptr<float>();
  const int32_t* activated_zero_data = activated_zero_view.const_data_ptr<int32_t>();
  at::parallel_for(0, unique * hidden, 1, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t slot = task / hidden;
      const int64_t channel = task % hidden;
      const int64_t first = offsets[slot];
      const int64_t last = offsets[slot + 1];
      const int8_t* weight = down_weight_data[slot] + channel * intermediate;
      for (int64_t position = first; position < last; ++position) {
        const int64_t assignment = group_assignments[position];
        const uint8_t* input = activated_q_data + assignment * intermediate;
        const int32_t dot = dot_u8_s8(input, weight, intermediate) -
            activated_zero_data[assignment] * down_sum_data[slot][channel];
        contribution_data[assignment * hidden + channel] =
            static_cast<float>(dot) * activated_scale_data[assignment] * down_scale_data[slot][channel];
      }
    }
  });

  auto output_view = output.narrow(0, 0, rows).narrow(1, 0, hidden);
  output_view.zero_();
  auto routing = routing_weights.to(at::kFloat).contiguous();
  const float* routing_data = routing.const_data_ptr<float>();
  auto* output_data = output_view.mutable_data_ptr<c10::BFloat16>();
  at::parallel_for(0, rows * hidden, 64, [&](int64_t begin, int64_t end) {
    for (int64_t index = begin; index < end; ++index) {
      const int64_t token = index / hidden;
      const int64_t channel = index % hidden;
      c10::BFloat16 accumulated = 0.0f;
      for (int64_t position = 0; position < top_k; ++position) {
        const int64_t assignment = token_order[token * top_k + position];
        const c10::BFloat16 expert_value = contribution_data[assignment * hidden + channel];
        const c10::BFloat16 route_value = routing_data[assignment];
        const c10::BFloat16 weighted = static_cast<float>(expert_value) * static_cast<float>(route_value);
        accumulated = static_cast<float>(accumulated) + static_cast<float>(weighted);
      }
      output_data[index] = accumulated;
    }
  });
  return output_view;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(sparseflow_native, m) {
  m.def(
      "grouped_moe(Tensor hidden_states, Tensor selected_experts, Tensor routing_weights, "
      "Tensor expert_ids, Tensor[] gate_up_weights, Tensor[] gate_up_scales, "
      "Tensor[] gate_up_row_sums, Tensor[] down_weights, Tensor[] down_scales, "
      "Tensor[] down_row_sums, Tensor plan_counts, Tensor plan_offsets, "
      "Tensor plan_assignments, Tensor plan_rows, Tensor plan_slots, Tensor plan_token_order, "
      "Tensor hidden_quantized, Tensor hidden_scales, Tensor hidden_zero_points, "
      "Tensor projected, Tensor activated, Tensor activated_quantized, Tensor activated_scales, "
      "Tensor activated_zero_points, Tensor contributions, Tensor output) -> Tensor");
}

TORCH_LIBRARY_IMPL(sparseflow_native, CPU, m) {
  m.impl("grouped_moe", &grouped_moe_cpu);
}

// [Main Dev]
