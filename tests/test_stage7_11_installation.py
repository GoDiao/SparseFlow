from pathlib import Path
import unittest

from benchmarks.run_stage7_11_installation_check import check_installation


class Stage711InstallationTest(unittest.TestCase):
    def test_non_clean_replay_cannot_pass_installation_gate(self):
        result = check_installation(Path(__file__).resolve().parents[1], clean_replay=False)
        self.assertFalse(result["passed"])
        self.assertFalse(result["checks"]["clean_replay_declared"])


if __name__ == "__main__":
    unittest.main()
