import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch

from sparseflow.cli import main
from sparseflow.int8_container import (
    ALIGNMENT,
    Int8ExpertIndex,
    convert_experts_int8,
    dequantize_part,
)


def write_fixture(root: Path) -> dict[tuple[int, str], torch.Tensor]:
    (root / "config.json").write_text(
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
    tensors = {
        (0, "gate_up_proj"): torch.tensor(
            [
                [[0.0, 1.0, -2.0], [3.0, -4.0, 5.0], [0.5, -0.5, 0.25], [7.0, 8.0, 9.0]],
                [[-1.0, 2.0, 3.0], [4.0, 5.0, -6.0], [0.0, 0.0, 0.0], [9.0, -8.0, 7.0]],
            ],
            dtype=torch.bfloat16,
        ),
        (0, "down_proj"): torch.tensor(
            [
                [[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [0.0, 0.0, 0.0, 0.0]],
                [[4.0, 3.0, 2.0, 1.0], [0.5, 0.25, -0.25, -0.5], [8.0, -8.0, 4.0, -4.0]],
            ],
            dtype=torch.bfloat16,
        ),
    }
    header = {}
    payload = bytearray()
    for (layer, part), tensor in tensors.items():
        name = f"model.language_model.layers.{layer}.mlp.experts.{part}"
        raw = tensor.view(torch.uint16).numpy().tobytes()
        start = len(payload)
        payload.extend(raw)
        header[name] = {
            "dtype": "BF16",
            "shape": list(tensor.shape),
            "data_offsets": [start, len(payload)],
        }
    encoded = json.dumps(header).encode("utf-8")
    (root / "model.safetensors").write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + payload
    )
    return tensors


class Int8ContainerTest(unittest.TestCase):
    def test_convert_locate_dequantize_and_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            output = root / "int8"
            model.mkdir()
            source = write_fixture(model)

            result = convert_experts_int8(model, output)
            self.assertEqual(result["experts"], 2)
            self.assertEqual(result["converted_layers"], 1)
            index = Int8ExpertIndex.from_dir(output)
            location = index.locate(0, 1)
            self.assertEqual(tuple(part.part for part in location.parts), ("gate_up_proj", "down_proj"))
            for part in location.parts:
                self.assertEqual(part.data_offset % ALIGNMENT, 0)
                self.assertEqual(part.scale_offset % ALIGNMENT, 0)
                restored = dequantize_part(part, index.read(0, 1)[part.part], torch)
                expected = source[(0, part.part)][1].float()
                self.assertTrue(torch.allclose(restored, expected, atol=0.08, rtol=0.02))

            resumed = convert_experts_int8(model, output, resume=True)
            self.assertEqual(resumed["converted_layers"], 0)
            self.assertEqual(resumed["resumed_layers"], 1)

    def test_cli_converts_selected_layers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            output = root / "int8"
            report = root / "report.json"
            model.mkdir()
            write_fixture(model)
            stdout = StringIO()
            with redirect_stdout(stdout):
                exit_code = main(
                    [
                        "int8-convert",
                        str(model),
                        "--output",
                        str(output),
                        "--layers",
                        "0",
                        "--threads",
                        "1",
                        "--report",
                        str(report),
                        "--json",
                    ]
                )
            self.assertEqual(exit_code, 0)
            self.assertEqual(json.loads(stdout.getvalue())["format_id"], "canonical-int8-v1")
            self.assertEqual(json.loads(report.read_text())["experts"], 2)


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
