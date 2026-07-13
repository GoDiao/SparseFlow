import json
import struct
import tempfile
import unittest
from pathlib import Path

from sparseflow.analyze import analyze_model
from sparseflow.plan import GB, build_plan


def write_shard(path: Path, tensors):
    offset = 0
    header = {}
    payload = b""
    for name, size, shape in tensors:
        header[name] = {
            "dtype": "U8",
            "shape": shape,
            "data_offsets": [offset, offset + size],
        }
        payload += b"\0" * size
        offset += size
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


class AnalyzePlanTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name)
        (self.model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "hidden_size": 16,
                        "num_hidden_layers": 2,
                        "num_experts": 4,
                        "num_experts_per_tok": 2,
                        "max_position_embeddings": 128,
                    },
                }
            ),
            encoding="utf-8",
        )
        write_shard(
            self.model / "model.safetensors",
            [
                ("model.language_model.embed_tokens.weight", 100, [10, 10]),
                ("lm_head.weight", 100, [10, 10]),
                ("model.language_model.layers.0.linear_attn.out_proj.weight", 40, [5, 8]),
                ("model.language_model.layers.0.mlp.gate.weight", 16, [4, 4]),
                ("model.language_model.layers.0.mlp.shared_expert.up_proj.weight", 24, [3, 8]),
                ("model.language_model.layers.0.mlp.experts.gate_up_proj", 80, [4, 20]),
                ("model.language_model.layers.0.mlp.experts.down_proj", 40, [4, 10]),
                ("model.language_model.layers.1.mlp.experts.gate_up_proj", 80, [4, 20]),
                ("model.language_model.layers.1.mlp.experts.down_proj", 40, [4, 10]),
                ("model.visual.patch_embed.weight", 32, [2, 16]),
            ],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_analyzes_qwen_fused_expert_layout(self):
        result = analyze_model(self.model)
        footprint = result["footprint"]

        self.assertEqual(result["model"]["model_type"], "qwen3_5_moe")
        self.assertEqual(footprint["category_bytes"]["routed_experts"], 240)
        self.assertEqual(footprint["category_bytes"]["embed_lm_head"], 200)
        self.assertEqual(footprint["category_bytes"]["routers"], 16)
        self.assertEqual(footprint["category_bytes"]["shared_experts"], 24)
        self.assertEqual(footprint["typical_layer_total_expert_bytes"], 120)
        self.assertEqual(footprint["typical_expert_bytes"], 30)
        self.assertEqual(footprint["per_layer_cache_slot_set_bytes"], 60)
        self.assertEqual(footprint["cold_expert_read_per_token_bytes"], 120)

    def test_plan_uses_one_slot_per_layer_cost(self):
        result = build_plan(
            self.model,
            ram_gb=2,
            ctx=16,
            reserve_gb=0,
            available_memory=4 * GB,
        )

        ram = result["tiers"]["ram"]
        self.assertGreater(ram["cache_slots_per_layer"], 1)
        self.assertEqual(result["warnings"], [])


if __name__ == "__main__":
    unittest.main()
