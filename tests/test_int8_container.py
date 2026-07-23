import json
import struct
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import torch

from sparseflow.cli import main
from sparseflow.cache import ExpertCache
from sparseflow.int8_container import (
    ALIGNMENT,
    Int8ExpertIndex,
    build_int8_execution_metadata,
    convert_experts_int8,
    dequantize_part,
)
from sparseflow.int8_provider import (
    Int8ResidentExpertProvider,
    Int8StreamingExpertProvider,
)
from sparseflow.loader import ShardReader


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

            manifest_before = (output / "manifest.json").read_bytes()
            resumed = convert_experts_int8(model, output, resume=True)
            self.assertEqual(resumed["converted_layers"], 0)
            self.assertEqual(resumed["resumed_layers"], 1)
            self.assertEqual((output / "manifest.json").read_bytes(), manifest_before)

            resident = Int8ResidentExpertProvider(output, torch)
            reader = ShardReader()
            streaming = Int8StreamingExpertProvider(
                output,
                ExpertCache(max_bytes=location.nbytes),
                reader,
                torch,
            )
            try:
                for expert in (0, 1, 0):
                    left = resident.get(0, expert)
                    right = streaming.get(0, expert)
                    for part_name in ("gate_up_proj", "down_proj"):
                        self.assertTrue(torch.equal(left[part_name], right[part_name]))
                self.assertEqual(streaming.counters()["demand_requests"], 3)
                self.assertGreater(streaming.counters()["demand_misses"], 0)
                self.assertLessEqual(streaming.counters()["cached_bytes"], location.nbytes)
            finally:
                resident.close()
                streaming.close()
                reader.close()

    def test_builds_resumable_offline_row_sums_sidecar(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            output = root / "int8"
            model.mkdir()
            write_fixture(model)
            convert_experts_int8(model, output)

            result = build_int8_execution_metadata(output)
            self.assertEqual(result["format_id"], "canonical-int8-exec-v1")
            self.assertEqual(result["entries"], 4)
            index = Int8ExpertIndex.from_dir(output)
            self.assertTrue(index.has_offline_row_sums)
            location = index.locate(0, 1)
            payload = index.read(0, 1)
            for part in location.parts:
                quantized = torch.frombuffer(
                    bytearray(payload[part.part]["data"]), dtype=torch.int8
                ).reshape(part.shape)
                expected = quantized.sum(dim=1, dtype=torch.int32)
                actual = torch.frombuffer(
                    bytearray(index.read_row_sums(0, 1, part.part)),
                    dtype=torch.int32,
                )
                self.assertTrue(torch.equal(actual, expected))

            manifest_before = (output / "execution-manifest.json").read_bytes()
            resumed = build_int8_execution_metadata(output, resume=True)
            self.assertEqual(resumed["converted_layers"], 0)
            self.assertEqual(resumed["resumed_layers"], 1)
            self.assertEqual(
                (output / "execution-manifest.json").read_bytes(),
                manifest_before,
            )

            native = Int8ResidentExpertProvider(output, torch, native=True)
            try:
                weights = native.get(0, 1)["gate_up_proj"]
                self.assertTrue(
                    torch.equal(
                        weights["gate_up_proj"]["row_sums"],
                        torch.frombuffer(
                            bytearray(index.read_row_sums(0, 1, "gate_up_proj")),
                            dtype=torch.int32,
                        ),
                    )
                )
                self.assertTrue(native.storage_report()["offline_row_sums"])
            finally:
                native.close()

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

    def test_streaming_provider_reuses_shared_prefetch_lifecycle(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            model = root / "model"
            output = root / "int8"
            model.mkdir()
            write_fixture(model)
            convert_experts_int8(model, output)
            index = Int8ExpertIndex.from_dir(output)
            budget = index.locate(0, 0).nbytes * 2
            resident = Int8ResidentExpertProvider(output, torch)
            reader = ShardReader()
            streaming = Int8StreamingExpertProvider(
                output,
                ExpertCache(max_bytes=budget),
                reader,
                torch,
                prefetch_workers=2,
                coalesce_gap=0,
                prefetch_policy="current-route",
            )
            try:
                streaming.begin_forward(0, "prefill")
                streaming.prepare(0, (0, 1))
                for expert in (0, 1):
                    left = resident.get(0, expert)
                    right = streaming.get(0, expert)
                    for part in ("gate_up_proj", "down_proj"):
                        self.assertTrue(torch.equal(left[part], right[part]))
                streaming.finish_generation()
                counters = streaming.counters()
                prefetch = streaming.prefetch_stats()
                self.assertEqual(prefetch["submitted"], 2)
                self.assertEqual(prefetch["completed"], 1)
                self.assertEqual(counters["demand_requests"], 2)
                self.assertEqual(
                    counters["demand_requests"],
                    counters["demand_reuse_hits"]
                    + counters["demand_prefetch_served"]
                    + counters["demand_misses"],
                )
                self.assertLessEqual(counters["cached_bytes"], budget)
            finally:
                resident.close()
                streaming.close()
                reader.close()


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
