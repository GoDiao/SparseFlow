import json
import struct
import tempfile
import unittest
from pathlib import Path

from sparseflow.benchmark import generate_trace, load_trace, parse_byte_budgets, run_expert_benchmark
from sparseflow.cache import ExpertCache
from sparseflow.loader import ShardReader
from sparseflow.locator import ExpertLocator
from sparseflow.trace import load_route_trace


def write_shard(path: Path, tensors):
    offset = 0
    header = {}
    payload = bytearray()
    for name, size, shape in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        payload.extend(bytes((offset + i) % 251 for i in range(size)))
        offset += size
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class ExpertCacheTest(unittest.TestCase):
    def test_per_layer_lru_and_stats(self):
        cache = ExpertCache(capacity_per_layer=1)

        def load(expert_id):
            return lambda: {"weight": bytes([expert_id])}

        cache.get_or_load(0, 0, load(0))
        cache.get_or_load(0, 0, load(0))
        cache.get_or_load(0, 1, load(1))
        cache.get_or_load(0, 0, load(0))
        cache.get_or_load(1, 0, load(0))

        self.assertEqual(cache.stats.requests, 5)
        self.assertEqual(cache.stats.hits, 1)
        self.assertEqual(cache.stats.misses, 4)
        self.assertEqual(cache.stats.evictions, 2)
        self.assertEqual(cache.stats.loaded_bytes, 4)
        self.assertEqual(cache.entries, 2)
        self.assertEqual(cache.cached_bytes, 2)

    def test_shard_reader_reuses_file_descriptors(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "shard.bin"
            path.write_bytes(b"abcdefgh")
            with ShardReader() as reader:
                self.assertEqual(reader.read(path, 1, 3), b"bcd")
                self.assertEqual(reader.read(path, 2, 2), b"cd")
                self.assertEqual(reader.read_calls, 2)
                self.assertEqual(reader.read_bytes, 5)
                self.assertEqual(len(reader._fds), 1)

    def test_batch_read_coalesces_multiple_expert_slices(self):
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
            write_shard(
                model / "model.safetensors",
                [
                    (
                        "model.language_model.layers.0.mlp.experts.gate_up_proj",
                        2 * 3,
                        [2, 3],
                    ),
                    (
                        "model.language_model.layers.0.mlp.experts.down_proj",
                        2 * 1,
                        [2, 1],
                    ),
                ],
            )
            locator = ExpertLocator(model)
            locations = [locator.locate(0, 0), locator.locate(0, 1)]
            with ShardReader() as reader:
                payloads = reader.read_locations(locations)
                stats = reader.last_batch_stats
            self.assertEqual(set(payloads), {(0, 0), (0, 1)})
            self.assertEqual(stats.logical_ranges, 4)
            self.assertEqual(stats.ranges, 1)
            self.assertEqual(stats.read_calls, 1)
            self.assertEqual(stats.useful_bytes, 8)
            self.assertEqual(stats.physical_bytes, 8)
            self.assertEqual(stats.wasted_bytes, 0)

    def test_global_byte_budget_evicts_across_layers(self):
        cache = ExpertCache(max_bytes=3)

        cache.get_or_load(0, 0, lambda: {"weight": b"aa"})
        cache.get_or_load(1, 0, lambda: {"weight": b"bb"})
        self.assertEqual(cache.cached_bytes, 2)
        cache.get_or_load(2, 0, lambda: {"weight": b"cc"})

        self.assertEqual(cache.cached_bytes, 2)
        self.assertIsNone(cache.lookup(0, 0))
        self.assertIsNotNone(cache.lookup(2, 0))

    def test_byte_budget_parser(self):
        self.assertEqual(parse_byte_budgets("512MiB,1GiB"), [512 * 1024**2, 1024**3])


class ExpertBenchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name)
        (self.model / "config.json").write_text(
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
        write_shard(
            self.model / "model.safetensors",
            [
                (
                    "model.language_model.layers.0.mlp.experts.gate_up_proj",
                    2 * 3,
                    [2, 3],
                ),
                (
                    "model.language_model.layers.0.mlp.experts.down_proj",
                    2 * 1,
                    [2, 1],
                ),
            ],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_generated_trace_and_capacity_sweep(self):
        trace = generate_trace([0], num_experts=2, tokens=3, top_k=1, mode="locality")
        result = run_expert_benchmark(self.model, [0, 1], trace)

        self.assertEqual(result["trace"]["requests"], 3)
        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["logical_bytes"], 12)
        self.assertGreaterEqual(result["results"][1]["hit_rate"], result["results"][0]["hit_rate"])

    def test_route_trace_json_replay_ignores_extra_metadata(self):
        trace_path = self.model / "route.json"
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": "qwen3_5_moe_route_trace",
                    "requests": [
                        {"forward": 0, "row": 0, "layer": 0, "expert": 1},
                        {"forward": 0, "row": 0, "layer": 0, "expert": 0},
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(load_trace(trace_path), [(0, 1), (0, 0)])

    def test_grouped_trace_preserves_forward_and_batch_union(self):
        trace_path = self.model / "route-v2.json"
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "forwards": [
                        {
                            "forward": 0,
                            "phase": "prefill",
                            "rows": [
                                {
                                    "row": 0,
                                    "token_position": 0,
                                    "token_id": 10,
                                    "layers": [{"layer": 0, "expert_ids": [0, 1]}],
                                },
                                {
                                    "row": 1,
                                    "token_position": 1,
                                    "token_id": 11,
                                    "layers": [{"layer": 0, "expert_ids": [1, 0]}],
                                },
                            ],
                        },
                        {
                            "forward": 1,
                            "phase": "decode",
                            "rows": [
                                {
                                    "row": 0,
                                    "token_position": 2,
                                    "token_id": 12,
                                    "layers": [{"layer": 0, "expert_ids": [1]}],
                                }
                            ],
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        trace = load_route_trace(trace_path)
        self.assertEqual(trace.raw_requests, 5)
        self.assertEqual(trace.phases, {"prefill": 4, "decode": 1})
        self.assertEqual(trace.batch_union_requests(), [(0, 0), (0, 1), (0, 1)])
        self.assertEqual([group.raw_requests for group in trace.replay_groups(True)], [4, 1])

        result = run_expert_benchmark(self.model, [0], trace, batch_union=True)
        self.assertEqual(result["trace"]["raw_requests"], 5)
        self.assertEqual(result["trace"]["effective_requests"], 3)
        self.assertEqual(result["trace"]["batch_union_deduped_requests"], 2)
        self.assertEqual(result["results"][0]["phase_metrics"]["prefill"]["requests"], 2)
        self.assertEqual(result["results"][0]["phase_metrics"]["decode"]["requests"], 1)
