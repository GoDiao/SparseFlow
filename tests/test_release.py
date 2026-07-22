import json
from pathlib import Path
import struct
import tempfile
import unittest

from sparseflow.cli import main
from sparseflow.release import PRESETS, apply_preset, container_identity, doctor, model_identity


def write_model(root: Path) -> Path:
    root.mkdir()
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "text_config": {
                    "model_type": "qwen3_5_moe_text",
                    "num_hidden_layers": 1,
                    "num_experts": 2,
                    "num_experts_per_tok": 1,
                },
            }
        ),
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        (root / name).write_text("{}" if name.endswith(".json") else "", encoding="utf-8")
    (root / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
    header = json.dumps(
        {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": {
                "dtype": "BF16",
                "shape": [2, 2, 2],
                "data_offsets": [0, 16],
            },
            "model.language_model.layers.0.mlp.experts.down_proj": {
                "dtype": "BF16",
                "shape": [2, 1, 2],
                "data_offsets": [16, 24],
            },
        }
    ).encode("utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(
        struct.pack("<Q", len(header)) + header + b"\0" * 24
    )
    return root


class ReleaseTest(unittest.TestCase):
    def test_public_presets_freeze_paths(self):
        stable = apply_preset("stable")
        low_memory = apply_preset("low-memory")
        batch = apply_preset("experimental-batch")
        self.assertEqual(stable["mode"], "resident")
        self.assertEqual(stable["native_dispatch"], "hybrid")
        self.assertEqual(low_memory["mode"], "streaming")
        self.assertEqual(low_memory["cache_policy"], "lru")
        self.assertEqual(low_memory["prefetch_workers"], 0)
        self.assertEqual(batch["native_dispatch"], "grouped")
        self.assertEqual(batch["batch_mode"], "fixed-cohort")
        self.assertFalse(low_memory["shared_streaming_batching"])

    def test_doctor_is_header_only_and_reports_missing_int8(self):
        with tempfile.TemporaryDirectory() as temp:
            model = write_model(Path(temp) / "model")
            result = doctor(model, preset="low-memory")
            self.assertFalse(result["ready"])
            self.assertEqual(result["model"]["payload_hash_mode"], "size-only")
            self.assertEqual(result["model"]["shards"], 1)
            checks = {item["id"]: item for item in result["checks"]}
            self.assertEqual(checks["safetensors_headers"]["status"], "pass")
            self.assertEqual(checks["int8_container"]["status"], "fail")

    def test_identity_and_cli_preset_are_serializable(self):
        with tempfile.TemporaryDirectory() as temp:
            model = write_model(Path(temp) / "model")
            identity = model_identity(model)
            self.assertEqual(identity["payload_hash_mode"], "size-only")
            container = Path(temp) / "container"
            container.mkdir()
            (container / "manifest.json").write_text("{}", encoding="utf-8")
            (container / "index.json").write_text("{}", encoding="utf-8")
            self.assertEqual(container_identity(container)["metadata_files"], ["index.json", "manifest.json"])
            self.assertEqual(main(["preset", "stable"]), 0)
            self.assertEqual(set(PRESETS), {"stable", "low-memory", "experimental-batch"})


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
