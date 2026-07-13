import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from sparseflow.cli import main
from sparseflow.loader import load_expert_raw
from sparseflow.locator import ExpertLocator, ExpertLocatorError


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


class ExpertLocatorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.model = Path(self.tmp.name)
        (self.model / "config.json").write_text(
            json.dumps(
                {
                    "model_type": "qwen3_5_moe",
                    "text_config": {
                        "model_type": "qwen3_5_moe_text",
                        "num_experts": 4,
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
                    4 * 2 * 3,
                    [4, 2, 3],
                ),
                (
                    "model.language_model.layers.0.mlp.experts.down_proj",
                    4 * 3 * 2,
                    [4, 3, 2],
                ),
            ],
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_locates_contiguous_slice_in_each_fused_part(self):
        locator = ExpertLocator(self.model)
        location = locator.locate(layer=0, expert_id=2)

        gate_up = location.part("gate_up_proj")
        down = location.part("down_proj")

        self.assertEqual(gate_up.expert_shape, (2, 3))
        self.assertEqual(gate_up.element_offset, 12)
        self.assertEqual(gate_up.element_count, 6)
        self.assertEqual(gate_up.nbytes, 6)
        self.assertEqual(down.expert_shape, (3, 2))
        self.assertEqual(down.element_offset, 12)
        self.assertEqual(down.element_count, 6)
        self.assertEqual(down.nbytes, 6)
        self.assertEqual(down.file_offset - gate_up.file_offset, 4 * 2 * 3)

    def test_rejects_out_of_range_expert(self):
        locator = ExpertLocator(self.model)
        with self.assertRaises(ExpertLocatorError):
            locator.locate(layer=0, expert_id=4)

    def test_rejects_non_contiguous_expert_axis(self):
        write_shard(
            self.model / "model-00002.safetensors",
            [
                (
                    "model.language_model.layers.1.mlp.experts.gate_up_proj",
                    2 * 4 * 3,
                    [2, 4, 3],
                ),
            ],
        )
        with self.assertRaises(ExpertLocatorError):
            ExpertLocator(self.model).locate(layer=1, expert_id=0)

    def test_loads_exact_ranges_and_cli_commands(self):
        loaded = load_expert_raw(self.model, layer=0, expert_id=2)
        self.assertEqual(loaded["total_bytes"], 12)
        self.assertEqual([part["bytes_read"] for part in loaded["parts"]], [6, 6])
        self.assertEqual(len(loaded["sha256"]), 64)

        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "expert-stat",
                        str(self.model),
                        "--layer",
                        "0",
                        "--expert",
                        "2",
                        "--json",
                    ]
                ),
                0,
            )
        stat = json.loads(output.getvalue())
        self.assertEqual(stat["parts"][0]["expert_id"], 2)

        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "expert-load",
                        str(self.model),
                        "--layer",
                        "0",
                        "--expert",
                        "2",
                        "--json",
                    ]
                ),
                0,
            )
        load = json.loads(output.getvalue())
        self.assertEqual(load["total_bytes"], 12)


if __name__ == "__main__":
    unittest.main()
