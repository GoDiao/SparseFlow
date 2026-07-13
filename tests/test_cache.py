import json
import struct
import tempfile
import unittest
from pathlib import Path

from sparseflow.benchmark import generate_trace, load_trace, run_expert_benchmark
from sparseflow.cache import ExpertCache
from sparseflow.locator import ExpertLocator


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
