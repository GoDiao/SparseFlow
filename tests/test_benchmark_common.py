import unittest

from benchmarks.common import delta, host_snapshot, process_snapshot


class BenchmarkCommonTest(unittest.TestCase):
    def test_process_and_host_snapshots_are_available(self):
        process = process_snapshot()
        host = host_snapshot()

        self.assertGreater(process["rss_bytes"], 0)
        self.assertGreaterEqual(process["peak_rss_bytes"], process["rss_bytes"])
        self.assertGreaterEqual(process["user_seconds"], 0)
        self.assertGreaterEqual(process["system_seconds"], 0)
        self.assertGreater(host["memory_total_bytes"], 0)
        self.assertGreater(host["memory_available_bytes"], 0)
        self.assertLessEqual(host["memory_available_bytes"], host["memory_total_bytes"])

    def test_delta_preserves_platform_metadata(self):
        result = delta(
            {"rss_bytes": 10, "platform_source": "old"},
            {"rss_bytes": 15, "platform_source": "windows-process-memory-info"},
        )
        self.assertEqual(result["rss_bytes"], 5)
        self.assertEqual(result["platform_source"], "windows-process-memory-info")


if __name__ == "__main__":
    unittest.main()
