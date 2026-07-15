import json
import struct
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - depends on the execution environment
    torch = None

from sparseflow.cache import ExpertCache
from sparseflow.expert_provider import (
    ExpertProvider,
    ResidentExpertProvider,
    StreamingExpertProvider,
)
from sparseflow.loader import ShardReader


def write_bf16_shard(path: Path, tensors):
    offset = 0
    header = {}
    payload = bytearray()
    for tensor_index, (name, shape) in enumerate(tensors):
        size = 1
        for dim in shape:
            size *= dim
        values = [0x3F80 + tensor_index + (index % 7) for index in range(size)]
        data = struct.pack(f"<{size}H", *values)
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        payload.extend(data)
        offset += len(data)
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)


def write_fixture(root: Path) -> None:
    (root / "config.json").write_text(
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
    tensors = []
    for layer in (0, 1):
        prefix = f"model.language_model.layers.{layer}.mlp.experts."
        tensors.extend(
            [
                (prefix + "gate_up_proj", (2, 6, 4)),
                (prefix + "down_proj", (2, 4, 3)),
            ]
        )
    write_bf16_shard(root / "model.safetensors", tensors)


@unittest.skipIf(torch is None, "torch is required for provider tests")
class ExpertProviderTest(unittest.TestCase):
    def test_resident_preloads_fused_tensors_and_performs_no_later_io(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            write_fixture(model)
            with ShardReader() as reader:
                provider = ResidentExpertProvider(model, reader, torch)
                self.assertIsInstance(provider, ExpertProvider)
                after_preload = (reader.read_calls, reader.read_bytes)
                weights = provider.get(0, 1)
                second = provider.get(1, 0)

                self.assertEqual(tuple(weights["gate_up_proj"].shape), (6, 4))
                self.assertEqual(tuple(second["down_proj"].shape), (4, 3))
                self.assertEqual((reader.read_calls, reader.read_bytes), after_preload)
                report = provider.storage_report()
                self.assertEqual(report["resident_layers"], 2)
                self.assertEqual(report["resident_experts"], 4)
                self.assertEqual(report["resident_buffers"], 4)
                self.assertEqual(report["preload_read_calls"], 4)
                self.assertEqual(report["reads_after_preload"], 0)
                self.assertEqual(report["bytes_after_preload"], 0)
                self.assertEqual(report["requests"], 2)

    def test_resident_and_streaming_return_exact_expert_tensors(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            write_fixture(model)
            resident_reader = ShardReader()
            streaming_reader = ShardReader()
            resident = ResidentExpertProvider(model, resident_reader, torch)
            streaming = StreamingExpertProvider(
                model,
                ExpertCache(capacity_per_layer=1),
                streaming_reader,
                torch,
            )
            try:
                self.assertIsInstance(streaming, ExpertProvider)
                for layer in (0, 1):
                    for expert_id in (0, 1):
                        left = resident.get(layer, expert_id)
                        right = streaming.get(layer, expert_id)
                        self.assertTrue(torch.equal(left["gate_up_proj"], right["gate_up_proj"]))
                        self.assertTrue(torch.equal(left["down_proj"], right["down_proj"]))
                self.assertEqual(resident.storage_report()["bytes_after_preload"], 0)
                self.assertGreater(streaming.storage_report()["reader_bytes"], 0)
            finally:
                resident.close()
                streaming.close()
                resident_reader.close()
                streaming_reader.close()

    def test_resident_close_invalidates_views_and_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            model = Path(temp)
            write_fixture(model)
            with ShardReader() as reader:
                provider = ResidentExpertProvider(model, reader, torch, layers=(0,))
                provider.close()
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    provider.get(0, 0)


if __name__ == "__main__":
    unittest.main()
