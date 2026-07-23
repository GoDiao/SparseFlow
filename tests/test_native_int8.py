import os
import shutil
import unittest

import torch

from sparseflow.moe_probe import run_routed_experts
from sparseflow.native_int8 import (
    _ensure_windows_msvc_environment,
    _native_cflags,
    load_native_int8,
    native_profile_snapshot,
    reference_dynamic_linear,
    run_native_expert,
    set_native_profile,
)
from sparseflow.native_moe import run_fused_native_moe, run_grouped_native_moe
from sparseflow.release import cpu_features


def has_native_requirements() -> bool:
    try:
        _ensure_windows_msvc_environment()
    except RuntimeError:
        return False
    compiler = shutil.which("cl") if os.name == "nt" else shutil.which("g++")
    return cpu_features()["avx512_vnni"] and compiler is not None and shutil.which("ninja") is not None


class NativeBuildConfigTest(unittest.TestCase):
    def test_native_flags_are_platform_specific(self):
        self.assertIn("/arch:AVX512", _native_cflags("nt"))
        self.assertIn("-mavx512vnni", _native_cflags("posix"))


@unittest.skipUnless(has_native_requirements(), "AVX-512 VNNI build environment required")
class NativeInt8Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        load_native_int8()

    def test_dynamic_linear_matches_scalar_quantization_reference(self):
        generator = torch.Generator().manual_seed(1234)
        input_tensor = torch.randn(3, 64, generator=generator)
        weight = torch.randint(-127, 128, (16, 64), generator=generator, dtype=torch.int8)
        scales = torch.rand(16, generator=generator) * 0.01 + 0.001
        row_sums = torch.ops.sparseflow_native.row_sums(weight)

        native = torch.ops.sparseflow_native.dynamic_linear(
            input_tensor, weight, scales, row_sums
        )
        reference = reference_dynamic_linear(
            input_tensor, weight, scales, row_sums, torch
        )
        self.assertTrue(torch.allclose(native, reference, atol=1e-5, rtol=1e-6))

    def test_profile_counters_are_opt_in_and_resettable(self):
        weight = torch.ones((8, 64), dtype=torch.int8)
        scales = torch.ones(8)
        row_sums = torch.ops.sparseflow_native.row_sums(weight)
        set_native_profile(True)
        torch.ops.sparseflow_native.dynamic_linear(
            torch.ones((1, 64)), weight, scales, row_sums
        )
        snapshot = native_profile_snapshot()
        self.assertEqual(snapshot["native_dynamic_linear_calls"], 1)
        self.assertGreater(snapshot["native_gemv_ns"], 0)
        set_native_profile(False)

    def test_fused_moe_is_deterministic_and_tracks_legacy_native_path(self):
        generator = torch.Generator().manual_seed(1234)

        def make_expert():
            gate_weight = torch.randint(
                -127, 128, (32, 64), generator=generator, dtype=torch.int8
            )
            down_weight = torch.randint(
                -127, 128, (64, 16), generator=generator, dtype=torch.int8
            )
            return {
                "native_int8": True,
                "gate_up_proj": {
                    "weight": gate_weight,
                    "scales": torch.rand(32, generator=generator) * 0.01 + 0.001,
                    "row_sums": torch.ops.sparseflow_native.row_sums(gate_weight),
                },
                "down_proj": {
                    "weight": down_weight,
                    "scales": torch.rand(64, generator=generator) * 0.01 + 0.001,
                    "row_sums": torch.ops.sparseflow_native.row_sums(down_weight),
                },
            }

        experts = {0: make_expert(), 1: make_expert()}

        class Provider:
            native = True

            def prepare(self, _layer, _expert_ids):
                pass

            def get(self, _layer, expert_id):
                return {"gate_up_proj": experts[expert_id], "down_proj": None}

        hidden = torch.randn((2, 64), generator=generator, dtype=torch.bfloat16)
        selected = torch.tensor([[1, 0], [0, 1]], dtype=torch.long)
        routing = torch.tensor([[0.4, 0.6], [0.7, 0.3]], dtype=torch.bfloat16)
        legacy = run_routed_experts(
            hidden,
            selected,
            routing,
            lambda expert_id: Provider().get(0, expert_id),
        )
        fused = run_fused_native_moe(hidden, selected, routing, Provider(), 0)
        repeated = run_fused_native_moe(hidden, selected, routing, Provider(), 0)
        self.assertTrue(torch.equal(fused, repeated))
        difference = (fused.float() - legacy.float()).abs()
        self.assertLess(float(difference.max()), 0.02)

    def test_grouped_moe_matches_fused_for_repeated_and_unique_routes(self):
        generator = torch.Generator().manual_seed(4321)

        def make_expert():
            gate_weight = torch.randint(
                -127, 128, (32, 64), generator=generator, dtype=torch.int8
            )
            down_weight = torch.randint(
                -127, 128, (64, 16), generator=generator, dtype=torch.int8
            )
            return {
                "native_int8": True,
                "gate_up_proj": {
                    "weight": gate_weight,
                    "scales": torch.rand(32, generator=generator) * 0.01 + 0.001,
                    "row_sums": torch.ops.sparseflow_native.row_sums(gate_weight),
                },
                "down_proj": {
                    "weight": down_weight,
                    "scales": torch.rand(64, generator=generator) * 0.01 + 0.001,
                    "row_sums": torch.ops.sparseflow_native.row_sums(down_weight),
                },
            }

        experts = {expert_id: make_expert() for expert_id in range(4)}

        class Provider:
            native = True

            def prepare(self, _layer, _expert_ids):
                pass

            def get(self, _layer, expert_id):
                return {"gate_up_proj": experts[expert_id], "down_proj": None}

        hidden = torch.randn((4, 64), generator=generator, dtype=torch.bfloat16)
        selected = torch.tensor(
            [[3, 1], [3, 1], [0, 2], [2, 0]], dtype=torch.long
        )
        routing = torch.tensor(
            [[0.4, 0.6], [0.7, 0.3], [0.2, 0.8], [0.9, 0.1]],
            dtype=torch.bfloat16,
        )
        provider = Provider()
        fused = run_fused_native_moe(hidden, selected, routing, provider, 0)
        grouped, workspace = run_grouped_native_moe(
            hidden, selected, routing, provider, 0
        )
        self.assertTrue(torch.equal(fused, grouped))
        self.assertGreater(workspace.allocated_bytes(), 0)
        grouped_again, same_workspace = run_grouped_native_moe(
            hidden, selected, routing, provider, 0, workspace=workspace
        )
        self.assertIs(same_workspace, workspace)
        self.assertTrue(torch.equal(grouped, grouped_again))


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
