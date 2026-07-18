import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

try:
    import torch
except ImportError:  # pragma: no cover - depends on the execution environment
    torch = None

from sparseflow.cache import ExpertCache
from sparseflow.cache_policy import make_cache_policy
from sparseflow.loader import ShardReader
from sparseflow.moe_probe import (
    CachedExpertProvider,
    compare_expert_paths,
    compare_moe_cache_paths,
    compare_moe_paths,
    run_routed_experts,
)
from sparseflow.moe_runtime import compare_multilayer_moe_paths
from sparseflow.cli import main


def write_bf16_shard(path: Path, tensors):
    offset = 0
    header = {}
    payload = bytearray()
    one = struct.pack("<H", 0x3F80)
    for name, shape in tensors:
        size = 1
        for dim in shape:
            size *= dim
        nbytes = size * 2
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        payload.extend(one * size)
        offset += nbytes
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


@unittest.skipIf(torch is None, "torch is required for the kernel probe")
class MoeProbeTest(unittest.TestCase):
    def test_routed_dispatch_matches_transformers_topk_then_token_order(self):
        hidden = torch.tensor(
            [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]],
            dtype=torch.bfloat16,
        )
        selected = torch.tensor([[1, 0], [0, 1], [1, 0]], dtype=torch.long)
        routing = torch.ones(3, 2, dtype=torch.bfloat16)
        observed = {}

        def record_kernel(current_state, gate_up_proj, _down_proj):
            observed[int(gate_up_proj)] = current_state[:, 0].tolist()
            return current_state

        def load(expert_id):
            return {"gate_up_proj": expert_id, "down_proj": expert_id}

        with patch("sparseflow.moe_probe.run_expert_kernel", side_effect=record_kernel):
            run_routed_experts(hidden, selected, routing, load)

        # Transformers constructs [expert, top_k, token] masks, so each
        # expert's rows are grouped by top-k position before token index.
        self.assertEqual(observed[0], [1.0, 0.0, 2.0])
        self.assertEqual(observed[1], [0.0, 2.0, 1.0])

    def test_single_token_fast_path_keeps_sorted_expert_accumulation(self):
        hidden = torch.tensor([[1.0, 2.0]], dtype=torch.bfloat16)
        selected = torch.tensor([[3, 1, 2]], dtype=torch.long)
        routing = torch.tensor([[0.5, 0.25, 0.125]], dtype=torch.bfloat16)
        loaded = []

        def load(expert_id):
            loaded.append(expert_id)
            return {"gate_up_proj": expert_id, "down_proj": expert_id}

        def kernel(_hidden, gate_up_proj, _down_proj):
            return torch.full((1, 2), float(gate_up_proj), dtype=torch.bfloat16)

        with patch("sparseflow.moe_probe.run_expert_kernel", side_effect=kernel):
            output = run_routed_experts(hidden, selected, routing, load)

        expected = torch.zeros_like(hidden)
        for expert_id, position in ((1, 1), (2, 2), (3, 0)):
            contribution = torch.full_like(hidden, float(expert_id))
            expected.add_((contribution * routing[0, position, None]).to(expected.dtype))
        self.assertEqual(loaded, [1, 2, 3])
        self.assertTrue(torch.equal(output, expected))

    def test_resident_and_streaming_use_the_same_kernel(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "text_config": {
                            "model_type": "qwen3_5_moe_text",
                            "num_hidden_layers": 1,
                            "num_experts": 2,
                            "hidden_size": 4,
                            "moe_intermediate_size": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_bf16_shard(
                model / "model.safetensors",
                [
                    ("model.language_model.layers.0.mlp.experts.gate_up_proj", (2, 6, 4)),
                    ("model.language_model.layers.0.mlp.experts.down_proj", (2, 4, 3)),
                ],
            )
            result = compare_expert_paths(model, layer=0, expert_id=1, rows=2)
            comparison = result["resident_vs_streaming"]
            self.assertTrue(comparison["exact_equal"])
            self.assertEqual(comparison["max_abs_error"], 0.0)
            self.assertEqual(result["parts"]["gate_up_proj"]["nbytes"], 48)
            self.assertEqual(result["parts"]["down_proj"]["nbytes"], 24)

    def test_complete_moe_resident_and_streaming_paths_match(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "text_config": {
                            "model_type": "qwen3_5_moe_text",
                            "num_hidden_layers": 1,
                            "num_experts": 2,
                            "num_experts_per_tok": 2,
                            "hidden_size": 4,
                            "moe_intermediate_size": 3,
                            "shared_expert_intermediate_size": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_bf16_shard(
                model / "model.safetensors",
                [
                    ("model.language_model.layers.0.mlp.gate.weight", (2, 4)),
                    ("model.language_model.layers.0.mlp.experts.gate_up_proj", (2, 6, 4)),
                    ("model.language_model.layers.0.mlp.experts.down_proj", (2, 4, 3)),
                    ("model.language_model.layers.0.mlp.shared_expert.gate_proj.weight", (3, 4)),
                    ("model.language_model.layers.0.mlp.shared_expert.up_proj.weight", (3, 4)),
                    ("model.language_model.layers.0.mlp.shared_expert.down_proj.weight", (4, 3)),
                    ("model.language_model.layers.0.mlp.shared_expert_gate.weight", (1, 4)),
                ],
            )

            result = compare_moe_paths(model, layer=0, rows=2, seed=1234)
            comparison = result["comparison"]
            for name in ("selected_experts", "routing_weights", "routed_output", "shared_output", "final_output"):
                self.assertTrue(comparison[name]["exact_equal"], name)
                self.assertEqual(comparison[name]["max_abs_error"], 0.0)
            self.assertEqual(result["resident_storage"]["read_bytes"], 144)
            self.assertEqual(result["streaming_storage"]["expert_count"], 2)
            self.assertEqual(result["streaming_storage"]["read_calls"], 4)
            self.assertEqual(result["streaming_storage"]["read_bytes"], 144)

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "expert-moe-check",
                            str(model),
                            "--layer",
                            "0",
                            "--rows",
                            "2",
                            "--json",
                        ]
                    ),
                    0,
                )
            cli_result = json.loads(output.getvalue())
            self.assertEqual(cli_result["kind"], "qwen3_5_moe_layer_correctness")
            self.assertTrue(cli_result["comparison"]["final_output"]["exact_equal"])

            cached = compare_moe_cache_paths(
                model,
                layer=0,
                rows=2,
                forwards=4,
                repeats=2,
                cache_slots=2,
            )
            self.assertTrue(cached["correctness"]["all_exact_equal"])
            self.assertEqual(cached["cache"]["requests"], 8)
            self.assertEqual(cached["cache"]["hits"], 6)
            self.assertEqual(cached["cache"]["misses"], 2)
            self.assertEqual(cached["cache"]["evictions"], 0)
            self.assertEqual(cached["cache"]["loaded_bytes"], 144)
            self.assertEqual(cached["streaming_storage"]["read_bytes"], 144)
            self.assertEqual(cached["cache"]["cached_bytes"], 144)
            self.assertTrue(all(cached["invariants"].values()))

            prefetched = compare_moe_cache_paths(
                model,
                layer=0,
                rows=2,
                forwards=4,
                repeats=2,
                cache_slots=2,
                prefetch_workers=2,
                coalesce_gap=0,
            )
            self.assertTrue(prefetched["correctness"]["all_exact_equal"])
            self.assertEqual(prefetched["cache"]["hits"], 7)
            self.assertEqual(prefetched["cache"]["misses"], 1)
            self.assertEqual(prefetched["prefetch"]["submitted"], 2)
            self.assertEqual(prefetched["prefetch"]["completed"], 1)
            self.assertGreaterEqual(prefetched["prefetch"]["waits"], 1)
            self.assertEqual(prefetched["prefetch"]["logical_ranges"], 4)
            self.assertEqual(prefetched["prefetch"]["coalesced_ranges"], 1)
            self.assertEqual(prefetched["prefetch"]["useful_bytes"], 144)
            self.assertEqual(prefetched["prefetch"]["physical_bytes"], 144)

            evicted = compare_moe_cache_paths(
                model,
                layer=0,
                rows=2,
                forwards=2,
                repeats=1,
                cache_slots=1,
            )
            self.assertTrue(evicted["correctness"]["all_exact_equal"])
            self.assertEqual(evicted["cache"]["requests"], 4)
            self.assertEqual(evicted["cache"]["hits"], 0)
            self.assertEqual(evicted["cache"]["misses"], 4)
            self.assertEqual(evicted["cache"]["evictions"], 3)
            self.assertEqual(evicted["cache"]["loaded_bytes"], 288)
            self.assertEqual(evicted["streaming_storage"]["read_bytes"], 288)
            self.assertEqual(evicted["cache"]["cached_bytes"], 72)
            self.assertTrue(all(evicted["invariants"].values()))

            budgeted = compare_moe_cache_paths(
                model,
                layer=0,
                rows=2,
                forwards=2,
                repeats=1,
                cache_slots=None,
                cache_bytes=72,
            )
            self.assertEqual(budgeted["cache_policy"]["max_bytes"], 72)
            self.assertEqual(budgeted["cache"]["cached_bytes"], 72)
            self.assertTrue(budgeted["invariants"]["cached_bytes_within_budget"])

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "expert-moe-cache-check",
                            str(model),
                            "--layer",
                            "0",
                            "--rows",
                            "2",
                            "--forwards",
                            "2",
                            "--repeats",
                            "2",
                            "--cache-slots",
                            "2",
                            "--json",
                        ]
                    ),
                    0,
                )
            cache_cli_result = json.loads(output.getvalue())
            self.assertEqual(cache_cli_result["kind"], "qwen3_5_moe_cache_correctness")
            self.assertTrue(cache_cli_result["correctness"]["all_exact_equal"])

    def test_cached_provider_preserves_per_layer_entries(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "text_config": {
                            "model_type": "qwen3_5_moe_text",
                            "num_hidden_layers": 2,
                            "num_experts": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_bf16_shard(
                model / "model.safetensors",
                [
                    ("model.language_model.layers.0.mlp.experts.gate_up_proj", (2, 6, 4)),
                    ("model.language_model.layers.0.mlp.experts.down_proj", (2, 4, 3)),
                    ("model.language_model.layers.1.mlp.experts.gate_up_proj", (2, 6, 4)),
                    ("model.language_model.layers.1.mlp.experts.down_proj", (2, 4, 3)),
                ],
            )
            cache = ExpertCache(capacity_per_layer=1)
            with ShardReader() as reader:
                provider = CachedExpertProvider(model, cache, reader, torch)
                self.assertEqual(provider.get(0, 0)["gate_up_proj"].shape, (6, 4))
                self.assertEqual(provider.get(1, 0)["gate_up_proj"].shape, (6, 4))
            self.assertEqual(cache.stats.requests, 2)
            self.assertEqual(cache.stats.misses, 2)
            self.assertEqual(cache.entries, 2)
            self.assertEqual(set(cache.cached_keys()), {(0, 0), (1, 0)})

    def test_two_layer_moe_runtime_with_prefetch_matches_resident(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "text_config": {
                            "model_type": "qwen3_5_moe_text",
                            "num_hidden_layers": 2,
                            "num_experts": 2,
                            "num_experts_per_tok": 2,
                            "hidden_size": 4,
                        },
                    }
                ),
                encoding="utf-8",
            )
            tensors = []
            for layer in (0, 1):
                prefix = f"model.language_model.layers.{layer}.mlp."
                tensors.extend(
                    [
                        (prefix + "gate.weight", (2, 4)),
                        (prefix + "experts.gate_up_proj", (2, 6, 4)),
                        (prefix + "experts.down_proj", (2, 4, 3)),
                        (prefix + "shared_expert.gate_proj.weight", (3, 4)),
                        (prefix + "shared_expert.up_proj.weight", (3, 4)),
                        (prefix + "shared_expert.down_proj.weight", (4, 3)),
                        (prefix + "shared_expert_gate.weight", (1, 4)),
                    ]
                )
            write_bf16_shard(model / "model.safetensors", tensors)

            result = compare_multilayer_moe_paths(
                model,
                layers=(0, 1),
                rows=2,
                cache_slots=2,
                prefetch_workers=2,
                coalesce_gap=0,
            )
            self.assertTrue(result["correctness"]["all_exact_equal"])
            self.assertEqual(len(result["per_layer"]), 2)
            self.assertTrue(all(result["invariants"].values()))
            self.assertEqual(result["prefetch"]["completed"], 2)
            self.assertEqual(result["prefetch"]["logical_ranges"], 8)
            self.assertEqual(result["prefetch"]["coalesced_ranges"], 2)
            self.assertEqual(
                result["streaming_storage"]["loaded_bytes"],
                result["streaming_storage"]["routed_read_bytes"],
            )

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "moe-multi-check",
                            str(model),
                            "--layers",
                            "0-1",
                            "--rows",
                            "2",
                            "--cache-slots",
                            "2",
                            "--prefetch-workers",
                            "2",
                            "--json",
                        ]
                    ),
                    0,
                )
            multi_cli_result = json.loads(output.getvalue())
            self.assertEqual(multi_cli_result["kind"], "qwen3_5_moe_multilayer_correctness")
            self.assertTrue(multi_cli_result["correctness"]["all_exact_equal"])

    def test_prefetch_failure_is_reported_and_does_not_leave_inflight_state(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "text_config": {
                            "model_type": "qwen3_5_moe_text",
                            "num_hidden_layers": 1,
                            "num_experts": 1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_bf16_shard(
                model / "model.safetensors",
                [
                    ("model.language_model.layers.0.mlp.experts.gate_up_proj", (1, 6, 4)),
                    ("model.language_model.layers.0.mlp.experts.down_proj", (1, 4, 3)),
                ],
            )
            cache = ExpertCache(capacity_per_layer=1)
            with ShardReader() as reader:
                provider = CachedExpertProvider(model, cache, reader, torch, prefetch_workers=1)

                def fail(*_args, **_kwargs):
                    raise OSError("synthetic prefetch failure")

                reader.read_locations = fail
                provider.prefetch(0, [0])
                with self.assertRaises(OSError):
                    provider.get(0, 0)
                stats = provider.prefetch_stats()
                self.assertEqual(stats["failed"], 1)
                self.assertEqual(stats["completed"], 0)
                self.assertEqual(provider._inflight, {})
                provider.close()

    def test_current_route_prefetch_reuses_transient_payload_when_admission_rejects(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            (model / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5_moe",
                        "text_config": {
                            "model_type": "qwen3_5_moe_text",
                            "num_hidden_layers": 1,
                            "num_experts": 2,
                        },
                    }
                ),
                encoding="utf-8",
            )
            write_bf16_shard(
                model / "model.safetensors",
                [
                    ("model.language_model.layers.0.mlp.experts.gate_up_proj", (2, 6, 4)),
                    ("model.language_model.layers.0.mlp.experts.down_proj", (2, 4, 3)),
                ],
            )
            policy = make_cache_policy("heat", max_hot_entries=0)
            cache = ExpertCache(max_bytes=1024, policy=policy)
            with ShardReader() as reader:
                provider = CachedExpertProvider(
                    model,
                    cache,
                    reader,
                    torch,
                    prefetch_workers=1,
                )
                provider.begin_forward(0, "prefill")
                selected = torch.tensor([[0, 1]], dtype=torch.long)
                provider.observe_routes(0, selected)
                provider.prepare(0, (0, 1))
                provider.get(0, 0)
                provider.get(0, 1)

                expected = sum(
                    provider.locator.locate(0, expert_id).nbytes
                    for expert_id in (0, 1)
                )
                snapshot = provider.snapshot()
                self.assertEqual(reader.read_bytes, expected)
                self.assertEqual(snapshot["demand_read_bytes"], 0)
                self.assertEqual(snapshot["demand_prefetch_served"], 2)
                self.assertEqual(snapshot["transient_prefetch_entries"], 0)
                self.assertEqual(cache.stats.admission_rejections, 2)
                provider.close()


if __name__ == "__main__":
    unittest.main()
