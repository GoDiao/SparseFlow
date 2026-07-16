import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import torch
from torch import nn

from sparseflow.cli import main
from sparseflow.text_runtime import (
    Qwen36TextRuntime,
    RouteAudit,
    SparseFlowQwenExperts,
    compare_generation_results,
    compare_sparseflow_policy_paths,
    compare_sparseflow_runtime_paths,
    compare_text_paths,
)
from sparseflow.telemetry import RuntimeTelemetry


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, *_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }

    def decode(self, token_ids, **_kwargs):
        return "tokens:" + ",".join(str(token) for token in token_ids)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **_kwargs):
        self.calls.append(
            {
                "input_shape": tuple(input_ids.shape),
                "mask_shape": tuple(attention_mask.shape),
                "has_past": past_key_values is not None,
                "use_cache": use_cache,
            }
        )
        vocab = 16
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab)
        token = 2 + len(self.calls) - 1
        logits[:, -1, token] = 1.0
        return types.SimpleNamespace(logits=logits, past_key_values=object())


class TextRuntimeTest(unittest.TestCase):
    def test_summary_telemetry_does_not_materialize_unique_experts(self):
        class Selected:
            shape = (1, 8)

            def numel(self):
                return 8

            def unique(self):
                raise AssertionError("summary telemetry must not compute unique experts")

        telemetry = RuntimeTelemetry("summary")
        telemetry.begin_forward(0, "decode", 3)
        telemetry.record_layer(0, Selected(), {}, {}, 1.0)
        result = telemetry.as_dict()

        self.assertEqual(result["summary"]["route_requests"], 8)
        self.assertIsNone(result["summary"]["unique_experts_sum"])
        self.assertEqual(result["records"], [])

    def test_layer_telemetry_preserves_forward_phase_and_counter_deltas(self):
        telemetry = RuntimeTelemetry("layer")
        telemetry.begin_forward(1, "decode", 12)
        telemetry.add_timing("router", 0.25)
        selected = torch.tensor([[1, 2]], dtype=torch.long)
        telemetry.record_layer(
            3,
            selected,
            {"reader_calls": 4, "reader_bytes": 10, "cache_hits": 1},
            {"reader_calls": 6, "reader_bytes": 30, "cache_hits": 2},
            1.5,
        )
        result = telemetry.as_dict()

        self.assertEqual(result["summary"]["decode_forwards"], 1)
        self.assertEqual(result["records"][0]["layer"], 3)
        self.assertEqual(result["records"][0]["provider"]["reader_calls"], 2)
        self.assertEqual(result["records"][0]["provider"]["reader_bytes"], 20)
        self.assertEqual(result["forwards"][0]["token_position"], 12)
        self.assertEqual(result["summary"]["timings_ms"]["router"], 0.25)
        self.assertEqual(result["records"][0]["timings_ms"]["router"], 0.25)

    def test_sparseflow_expert_module_uses_provider_for_shared_dispatch(self):
        class Provider:
            backend_id = "test"

            def __init__(self):
                self.prepared = []

            def prepare(self, layer, expert_ids):
                self.prepared.append((layer, expert_ids))

            def observe_routes(self, _layer, _selected_experts):
                pass

            def snapshot(self):
                return {}

            def get(self, _layer, expert_id):
                scale = float(expert_id + 1)
                return {
                    "gate_up_proj": torch.full((4, 2), scale, dtype=torch.bfloat16),
                    "down_proj": torch.full((2, 2), scale, dtype=torch.bfloat16),
                }

        provider = Provider()
        audit = RouteAudit()
        module = SparseFlowQwenExperts(3, provider, audit)
        hidden = torch.ones((2, 2), dtype=torch.bfloat16)
        selected = torch.tensor([[0], [1]], dtype=torch.long)
        routing = torch.ones((2, 1), dtype=torch.bfloat16)
        output = module(hidden, selected, routing)

        self.assertEqual(tuple(output.shape), (2, 2))
        self.assertEqual(provider.prepared, [(3, (0, 1))])
        self.assertEqual(audit.records[0]["layer"], 3)
        self.assertEqual(audit.records[0]["shape"], [2, 1])

    def test_prefill_decode_and_greedy_generation(self):
        model = FakeModel()
        runtime = Qwen36TextRuntime(
            model,
            FakeTokenizer(),
            torch,
            "/tmp/fake-qwen36",
            mode="resident",
        )
        result = runtime.greedy_generate(
            "hello",
            max_new_tokens=3,
            stop_on_eos=True,
            record_logit_fingerprints=True,
        )

        self.assertEqual(result["generated_ids"], [2, 3, 4])
        self.assertEqual(result["generated_tokens"], 3)
        self.assertEqual(result["text"], "tokens:2,3,4")
        self.assertGreaterEqual(result["prefill_seconds"], 0.0)
        self.assertEqual(len(result["decode_token_seconds"]), 2)
        self.assertEqual(len(result["logit_fingerprints"]), 3)
        self.assertEqual(
            [item["argmax"] for item in result["logit_fingerprints"]],
            [[2], [3], [4]],
        )
        self.assertEqual([call["has_past"] for call in model.calls], [False, True, True])
        self.assertEqual([call["input_shape"] for call in model.calls], [(1, 2), (1, 1), (1, 1)])
        self.assertTrue(all(call["use_cache"] for call in model.calls))

    def test_generation_comparison_requires_complete_external_equality(self):
        resident = {
            "input_ids": [1, 2],
            "generated_ids": [3, 4],
            "generated_tokens": 2,
            "text": "ok",
            "logit_fingerprints": [{"sha256": "same"}],
        }
        self.assertTrue(compare_generation_results(resident, dict(resident))["all_equal"])

        changed = dict(resident)
        changed["generated_ids"] = [3, 5]
        comparison = compare_generation_results(resident, changed)
        self.assertFalse(comparison["generated_ids_equal"])
        self.assertFalse(comparison["all_equal"])

        changed = dict(resident)
        changed["logit_fingerprints"] = [{"sha256": "different"}]
        comparison = compare_generation_results(resident, changed)
        self.assertFalse(comparison["logit_fingerprints_equal"])
        self.assertFalse(comparison["all_equal"])

    def test_text_check_cli_emits_comparison_and_uses_failure_exit_code(self):
        result = {
            "correctness": {"all_equal": True},
            "resident": {
                "generated_ids": [3],
                "text": "ok",
                "load_seconds": 1.0,
                "prefill_seconds": 2.0,
                "decode_seconds": 0.0,
            },
            "streaming": {
                "generated_ids": [3],
                "text": "ok",
                "load_seconds": 1.5,
                "prefill_seconds": 3.0,
                "decode_seconds": 0.0,
                "cache": {"requests": 2, "hits": 1, "misses": 1, "evictions": 0},
            },
        }
        output = StringIO()
        with patch("sparseflow.cli.compare_text_paths", return_value=result):
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "text-check",
                        "/tmp/model",
                        "--prompt",
                        "hello",
                    ]
                )
        self.assertEqual(exit_code, 0)
        self.assertIn("SparseFlow text-check", output.getvalue())

        result["correctness"]["all_equal"] = False
        with patch("sparseflow.cli.compare_text_paths", return_value=result):
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "text-check",
                        "/tmp/model",
                        "--prompt",
                        "hello",
                    ]
                )
        self.assertEqual(exit_code, 1)

    def test_text_check_pins_eager_arithmetic_for_both_paths(self):
        result = {
            "input_ids": [1],
            "generated_ids": [2],
            "generated_tokens": 1,
            "text": "ok",
            "logit_fingerprints": [{"sha256": "same"}],
        }
        with patch(
            "sparseflow.text_runtime._run_text_path",
            side_effect=[(dict(result), 1.0), (dict(result), 2.0)],
        ) as run_path:
            comparison = compare_text_paths(
                "/tmp/model",
                "hello",
                max_new_tokens=1,
            )

        self.assertTrue(comparison["correctness"]["all_equal"])
        self.assertEqual(comparison["experts_implementation"], "eager")
        self.assertEqual(run_path.call_count, 2)
        self.assertEqual(run_path.call_args_list[0].kwargs["mode"], "resident")
        self.assertEqual(run_path.call_args_list[1].kwargs["mode"], "streaming")
        self.assertEqual(
            [call.kwargs["experts_implementation"] for call in run_path.call_args_list],
            ["eager", "eager"],
        )
        self.assertEqual(run_path.call_args_list[0].kwargs["load_mode"], "transformers")
        self.assertEqual(run_path.call_args_list[1].kwargs["load_mode"], "transformers")

    def test_text_check_routes_streaming_through_memory_native_loader(self):
        result = {
            "input_ids": [1],
            "generated_ids": [2],
            "generated_tokens": 1,
            "text": "ok",
            "logit_fingerprints": [{"sha256": "same"}],
        }
        with patch(
            "sparseflow.text_runtime._run_text_path",
            side_effect=[(dict(result), 1.0), (dict(result), 2.0)],
        ) as run_path:
            comparison = compare_text_paths(
                "/tmp/model",
                "hello",
                max_new_tokens=1,
                streaming_load_mode="memory-native",
            )
        self.assertTrue(comparison["correctness"]["all_equal"])
        self.assertEqual(comparison["streaming_load_mode"], "memory-native")
        self.assertEqual(
            run_path.call_args_list[1].kwargs["load_mode"],
            "memory-native",
        )

    def test_stage72_runtime_check_pins_same_memory_native_runtime(self):
        identity = {
            "runtime_id": "same",
            "expert_module_id": "same",
            "dispatch_id": "same",
            "kernel_id": "same",
        }

        def result(backend):
            resident = backend == "sparseflow-resident"
            before = {"reader_calls": 80 if resident else 0, "reader_bytes": 400 if resident else 0}
            after = dict(before) if resident else {"reader_calls": 2, "reader_bytes": 20}
            storage = {
                "backend_id": backend,
                "resident_layers": 2 if resident else 0,
                "resident_experts": 4 if resident else 0,
                "resident_bytes": 400 if resident else 0,
                "preload_read_bytes": 400 if resident else 0,
            }
            if not resident:
                storage.update({"reader_bytes": 20, "reader_calls": 2})
            return {
                "input_ids": [1],
                "generated_ids": [2],
                "generated_tokens": 1,
                "text": "ok",
                "logit_fingerprints": [{"sha256": "same"}],
                "route_audit": [{"sha256": "route"}],
                "runtime_identity": identity,
                "provider_storage": storage,
                "storage_phases": {
                    "before_prefill": before,
                    "after_generation": after,
                },
                "loader": {
                    "expert_reader_calls_after_init": 80 if resident else 0,
                    "expert_reader_bytes_after_init": 400 if resident else 0,
                },
            }

        fake_locator = types.SimpleNamespace(
            layers=(0, 1),
            num_experts=2,
            fused_parts=lambda _layer: {
                "gate_up_proj": types.SimpleNamespace(nbytes=100),
                "down_proj": types.SimpleNamespace(nbytes=100),
            },
        )
        with patch("sparseflow.text_runtime.ExpertLocator", return_value=fake_locator):
            with patch(
                "sparseflow.text_runtime._run_text_path",
                side_effect=[
                    (result("sparseflow-resident"), 1.0),
                    (result("sparseflow-streaming"), 2.0),
                ],
            ) as run_path:
                comparison = compare_sparseflow_runtime_paths(
                    "/tmp/model",
                    "hello",
                    max_new_tokens=1,
                )

        self.assertTrue(comparison["all_invariants_pass"])
        self.assertEqual(run_path.call_args_list[0].kwargs["mode"], "resident")
        self.assertEqual(run_path.call_args_list[1].kwargs["mode"], "streaming")
        self.assertTrue(
            all(call.kwargs["load_mode"] == "memory-native" for call in run_path.call_args_list)
        )
        self.assertTrue(
            all(
                call.kwargs["experts_implementation"] == "eager"
                for call in run_path.call_args_list
            )
        )

    def test_runtime_check_cli_returns_invariant_status(self):
        result = {
            "all_invariants_pass": True,
            "runtime_identity": {"kernel_id": "same"},
            "correctness": {"all_equal": True},
            "invariants": {"same_runtime_identity": True},
            "resident": {
                "generated_ids": [2],
                "text": "ok",
                "provider_storage": {"resident_bytes": 100},
                "generation_expert_io": {"read_bytes": 0},
            },
            "streaming": {
                "generated_ids": [2],
                "text": "ok",
                "generation_expert_io": {"read_bytes": 20},
            },
        }
        with patch("sparseflow.cli.compare_sparseflow_runtime_paths", return_value=result):
            output = StringIO()
            with redirect_stdout(output):
                exit_code = main(
                    ["runtime-check", "/tmp/model", "--prompt", "hello"]
                )
        self.assertEqual(exit_code, 0)
        self.assertIn("Stage 7.2 runtime-check", output.getvalue())

    def test_stage73_policy_check_reuses_resident_reference_and_checks_accounting(self):
        base = {
            "input_ids": [1],
            "generated_ids": [2, 3],
            "generated_tokens": 2,
            "text": "ok",
            "logit_fingerprints": [{"sha256": "a"}, {"sha256": "b"}],
            "route_audit": [{"sha256": "r"}],
            "runtime_identity": {"kernel_id": "same"},
        }
        streaming = {
            **base,
            "provider_storage": {
                "cached_bytes": 4,
                "reader_bytes": 8,
                "demand_requests": 2,
                "demand_reuse_hits": 1,
                "demand_prefetch_served": 0,
                "demand_misses": 1,
            },
            "loader": {
                "expert_reader_calls_after_init": 0,
                "expert_reader_bytes_after_init": 0,
            },
            "prefetch": {"failed": 0},
        }
        with patch(
            "sparseflow.text_runtime._run_text_path",
            side_effect=[(dict(base), 1.0), (streaming, 2.0)],
        ) as run_path:
            result = compare_sparseflow_policy_paths(
                "/tmp/model",
                "hello",
                max_new_tokens=2,
                cache_bytes=16,
                variants=("S1",),
            )

        self.assertTrue(result["all_invariants_pass"])
        self.assertEqual(run_path.call_count, 2)
        self.assertEqual(run_path.call_args_list[0].kwargs["mode"], "resident")
        self.assertEqual(run_path.call_args_list[1].kwargs["mode"], "streaming")
        self.assertEqual(run_path.call_args_list[1].kwargs["cache_policy"], "lru")


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
