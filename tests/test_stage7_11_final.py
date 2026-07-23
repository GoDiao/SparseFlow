from pathlib import Path
import unittest

from benchmarks.run_stage7_11_environment import capture_environment
from benchmarks.verify_stage7_11_final import verify


class Stage711FinalVerificationTest(unittest.TestCase):
    def test_missing_evidence_is_not_run(self):
        root = Path(__file__).resolve().parents[1]
        result = verify(
            root=root,
            environment=None,
            doctor=None,
            cli=None,
            cli_verification=None,
            cli_performance=None,
            server=None,
            api_contract=None,
            installation=None,
        )
        self.assertEqual(result["overall_decision"], "NO-GO")
        self.assertEqual(result["gates"]["environment_gate"], "NOT-RUN")
        self.assertEqual(result["gates"]["doctor_gate"], "NOT-RUN")
        self.assertEqual(result["gates"]["api_contract_gate"], "NOT-RUN")
        self.assertEqual(result["gates"]["frontend_readiness_gate"], "NO-GO")

    def test_environment_capture_is_not_formal_result(self):
        root = Path(__file__).resolve().parents[1]
        result = capture_environment(root=root)
        self.assertEqual(result["kind"], "sparseflow_stage7_11_environment")
        self.assertFalse(result["scope"]["formal_result"])
        self.assertIn("git", result)

    def test_formal_server_gates_do_not_hide_a_failed_cell(self):
        root = Path(__file__).resolve().parents[1]
        server = {"gates": {"health_ready": True, "runtime_load_once": True}}
        result = verify(
            root=root,
            environment={"git": {"commit": "abc"}},
            doctor={"rows": [{"cache_bytes": 256 * 1024**2, "ready": True}]},
            cli={"samples": []},
            cli_verification={"verification_passed": True, "checks": {"repeat_correctness_exact": True, "memory_nonzero": True}},
            cli_performance={"passed": True},
            server=server,
            api_contract={"passed": True, "git": {"commit": "abc", "dirty": False}},
            installation={"passed": True},
        )
        self.assertEqual(result["gates"]["server_identity_gate"], "NO-GO")
        self.assertFalse(result["evidence"]["identity_consistent"])
        self.assertEqual(result["overall_decision"], "NO-GO")


if __name__ == "__main__":
    unittest.main()
