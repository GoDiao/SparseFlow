import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from sparseflow.cli import main
from sparseflow.memory_loader import build_memory_load_plan


def write_shard(path: Path, tensors):
    header = {}
    payload = bytearray()
    offset = 0
    for name, shape in tensors:
        elements = 1
        for dim in shape:
            elements *= dim
        nbytes = elements * 2
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + nbytes],
        }
        payload.extend(b"\x00" * nbytes)
        offset += nbytes
    encoded = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


class MemoryLoadPlanTest(unittest.TestCase):
    def make_model(self, root: Path, include_down: bool = True):
        (root / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "num_hidden_layers": 2,
                    },
                }
            ),
            encoding="utf-8",
        )
        tensors = [
            ("model.language_model.embed_tokens.weight", [8, 4]),
            ("model.language_model.layers.0.mlp.experts.gate_up_proj", [2, 6, 4]),
            ("model.language_model.layers.1.mlp.experts.gate_up_proj", [2, 6, 4]),
            ("model.visual.patch_embed.weight", [2, 2]),
            ("mtp.layers.0.mlp.experts.gate_up_proj", [2, 6, 4]),
            ("lm_head.weight", [8, 4]),
        ]
        if include_down:
            tensors.extend(
                [
                    ("model.language_model.layers.0.mlp.experts.down_proj", [2, 4, 3]),
                    ("model.language_model.layers.1.mlp.experts.down_proj", [2, 4, 3]),
                ]
            )
        write_shard(root / "model.safetensors", tensors)

    def test_qwen36_plan_maps_text_and_skips_non_text_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_model(root)
            plan = build_memory_load_plan(root)
            result = plan.as_dict(include_entries=False)

            self.assertTrue(result["header_only"])
            self.assertEqual(result["payload_bytes_read"], 0)
            self.assertEqual(result["tensor_counts"], {"resident": 2, "skip": 2, "stream": 4})
            self.assertEqual(result["reason_counts"]["routed_expert"], 4)
            self.assertEqual(result["reason_counts"]["mtp"], 1)
            self.assertEqual(result["reason_counts"]["vision"], 1)
            targets = {entry.source_name: entry.target_name for entry in plan.resident_entries}
            self.assertEqual(
                targets["model.language_model.embed_tokens.weight"],
                "model.embed_tokens.weight",
            )
            self.assertEqual(targets["lm_head.weight"], "lm_head.weight")

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["native-plan", str(root), "--json"]), 0)
            cli_result = json.loads(output.getvalue())
            self.assertEqual(cli_result["payload_bytes_read"], 0)
            self.assertEqual(cli_result["tensor_counts"]["stream"], 4)

    def test_plan_rejects_incomplete_expert_layers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_model(root, include_down=False)
            with self.assertRaisesRegex(ValueError, "missing parts"):
                build_memory_load_plan(root)


if __name__ == "__main__":
    unittest.main()

