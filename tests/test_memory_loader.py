import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

from sparseflow.cli import main
from sparseflow.memory_loader import build_memory_load_plan, prepare_qwen36_meta_text_model


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

    @unittest.skipIf(torch is None, "torch is required for meta model tests")
    def test_prepare_meta_model_replaces_experts_before_loading(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            self.make_model(root)
            plan = build_memory_load_plan(root)

            class Mlp(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.experts = nn.Linear(4, 4, bias=False, device="meta")

            class Layer(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.mlp = Mlp()

            class Body(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.embed_tokens = nn.Embedding(8, 4, device="meta")
                    self.layers = nn.ModuleList([Layer(), Layer()])

            class Model(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.model = Body()
                    self.lm_head = nn.Linear(4, 8, bias=False, device="meta")

            class EmptyExperts(nn.Module):
                def __init__(self, layer):
                    super().__init__()
                    self.layer = layer

            build = prepare_qwen36_meta_text_model(Model(), plan, EmptyExperts)
            result = build.as_dict()
            self.assertEqual(result["state_tensors"], 2)
            self.assertEqual(result["routed_expert_parameters"], 0)
            self.assertEqual(result["payload_bytes_read"], 0)
            self.assertTrue(all(parameter.is_meta for parameter in build.model.parameters()))
            self.assertEqual(
                [layer.mlp.experts.layer for layer in build.model.model.layers],
                [0, 1],
            )


if __name__ == "__main__":
    unittest.main()
