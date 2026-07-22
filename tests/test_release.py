import json
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import unittest

from sparseflow.cli import main
from sparseflow.release import (
    GIB,
    PRESETS,
    apply_preset,
    container_identity,
    doctor,
    evaluate_memory_admission,
    memory_snapshot,
    model_identity,
)


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

    def test_memory_admission_preset_budgets_and_cache_override(self):
        analysis = {
            "model": {
                "path": "/tmp/qwen36",
                "hidden_size": 2048,
                "num_hidden_layers": 40,
            },
            "footprint": {
                "dense_resident_bytes": 7 * GIB,
                "routed_expert_bytes": 60 * GIB,
            },
        }
        container = {"weight_bytes": 30 * GIB, "execution_bytes": 126 * 1024**2}

        stable_16 = evaluate_memory_admission(
            analysis,
            preset="stable",
            effective_config=apply_preset("stable"),
            container=container,
            available_ram_bytes=16 * GIB,
        )
        stable_128 = evaluate_memory_admission(
            analysis,
            preset="stable",
            effective_config=apply_preset("stable"),
            container=container,
            available_ram_bytes=128 * GIB,
        )
        low_16 = evaluate_memory_admission(
            analysis,
            preset="low-memory",
            effective_config=apply_preset("low-memory"),
            container=container,
            available_ram_bytes=16 * GIB,
        )
        self.assertEqual(stable_16["status"], "fail")
        self.assertEqual(stable_128["status"], "pass")
        self.assertIn(low_16["status"], {"pass", "warn"})
        self.assertEqual(
            low_16["components"]["streaming_cache_bytes"],
            4 * GIB,
        )

        low_8 = evaluate_memory_admission(
            analysis,
            preset="low-memory",
            effective_config=apply_preset("low-memory", cache_bytes=8 * GIB),
            container=container,
            available_ram_bytes=16 * GIB,
        )
        self.assertEqual(low_8["components"]["streaming_cache_bytes"], 8 * GIB)
        self.assertGreater(
            low_8["required_ram_bytes"], low_16["required_ram_bytes"]
        )

    def test_memory_snapshot_cgroup_override_and_missing_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            proc = root / "meminfo"
            proc.write_text(
                f"MemTotal:       {131072 * 1024} kB\n"
                f"MemAvailable:   {131072 * 1024} kB\n",
                encoding="utf-8",
            )
            cgroup = root / "cgroup"
            cgroup.mkdir()
            (cgroup / "memory.max").write_text(str(16 * GIB), encoding="utf-8")
            (cgroup / "memory.current").write_text(str(4 * GIB), encoding="utf-8")
            limited = memory_snapshot(proc_meminfo_path=proc, cgroup_root=cgroup)
            self.assertEqual(limited["source"], "cgroup")
            self.assertEqual(limited["available_ram_bytes"], 12 * GIB)

            override = memory_snapshot(
                override_bytes=8 * GIB,
                proc_meminfo_path=proc,
                cgroup_root=cgroup,
            )
            self.assertEqual(override["source"], "override")
            self.assertEqual(override["available_ram_bytes"], 8 * GIB)

            missing = memory_snapshot(
                proc_meminfo_path=root / "missing-meminfo",
                cgroup_root=root / "missing-cgroup",
            )
            self.assertEqual(missing["source"], "unknown")
            self.assertEqual(missing["available_ram_bytes"], 0)

    def test_base_cli_subprocess_has_no_runtime_import_dependency(self):
        with tempfile.TemporaryDirectory() as temp:
            model = write_model(Path(temp) / "model")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            env["PYTHONNOUSERSITE"] = "1"

            def run(*args: str) -> subprocess.CompletedProcess[str]:
                return subprocess.run(
                    [sys.executable, "-S", "-m", "sparseflow", *args],
                    env=env,
                    text=True,
                    capture_output=True,
                    check=False,
                )

            preset = run("preset", "stable", "--json")
            inspect = run("inspect", str(model), "--json")
            plan = run("plan", str(model), "--ram", "16", "--json")
            doctor_result = run("doctor", str(model), "--json")
            runtime = run(
                "run",
                str(model),
                "--preset",
                "stable",
                "--int8-container",
                str(Path(temp) / "missing-int8"),
                "--prompt",
                "test",
            )

            for result in (preset, inspect, plan, doctor_result):
                combined = result.stdout + result.stderr
                self.assertNotIn("Traceback", combined)
                self.assertNotIn("ModuleNotFoundError", combined)
            self.assertEqual(preset.returncode, 0)
            self.assertEqual(inspect.returncode, 0)
            self.assertEqual(plan.returncode, 0)
            self.assertNotEqual(doctor_result.returncode, 2)
            self.assertEqual(runtime.returncode, 2)
            self.assertIn("requires the optional runtime dependencies", runtime.stderr)
            self.assertNotIn("Traceback", runtime.stderr)


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
