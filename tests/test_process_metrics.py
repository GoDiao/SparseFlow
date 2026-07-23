import unittest

from sparseflow.process_metrics import (
    current_rss_bytes,
    host_memory_snapshot,
    peak_rss_bytes,
    process_snapshot,
)


class ProcessMetricsTest(unittest.TestCase):
    def test_process_snapshot_reports_nonzero_memory_and_semantics(self):
        snapshot = process_snapshot()

        self.assertGreater(snapshot["rss_bytes"], 0)
        self.assertGreaterEqual(snapshot["peak_rss_bytes"], snapshot["rss_bytes"])
        self.assertGreaterEqual(snapshot["private_bytes"], 0)
        self.assertIn("read_bytes_semantics", snapshot)
        self.assertTrue(snapshot["platform_source"])
        self.assertGreater(current_rss_bytes(), 0)
        self.assertGreaterEqual(peak_rss_bytes(), snapshot["peak_rss_bytes"])

    def test_host_memory_snapshot_reports_a_valid_budget(self):
        snapshot = host_memory_snapshot()

        self.assertGreater(snapshot["total_bytes"], 0)
        self.assertGreater(snapshot["available_bytes"], 0)
        self.assertLessEqual(snapshot["available_bytes"], snapshot["total_bytes"])
        self.assertTrue(snapshot["source"])


if __name__ == "__main__":
    unittest.main()
