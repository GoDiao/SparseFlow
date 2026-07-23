from pathlib import Path
import unittest

from benchmarks.run_stage7_11_installation_check import check_installation


class Stage711InstallationTest(unittest.TestCase):
    def test_non_clean_replay_cannot_pass_installation_gate(self):
        result = check_installation(Path(__file__).resolve().parents[1], clean_replay=False)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["clean_replay_declared"])

    def test_windows_entrypoints_keep_utf8_mode_enabled(self):
        root = Path(__file__).resolve().parents[1]
        for name in ("setup_windows.ps1", "use_runtime_windows.ps1"):
            content = (root / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn('$env:PYTHONUTF8 = "1"', content)


if __name__ == "__main__":
    unittest.main()
