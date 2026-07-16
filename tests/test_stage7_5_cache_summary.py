import unittest

from benchmarks.summarize_stage7_5_cache import recommendations


class Stage75CacheSummaryTest(unittest.TestCase):
    def test_recommendations_select_measured_winners(self):
        rows = []
        for variant, speed in (("C3-S1", 1.0), ("C3-S3", 0.9), ("C3-S4", 0.8)):
            rows.append(
                {
                    "tag": "core",
                    "variant": variant,
                    "cache_bytes": 1024,
                    "decode_tokens_per_second": speed,
                    "physical_bytes_per_decode_token": 10,
                    "rss_after_generation": 20,
                }
            )
        rows.extend(
            (
                {"tag": "cold", "variant": "C3-S1", "decode_tokens_per_second": 0.7, "ttft_seconds": 2},
                {"tag": "cold", "variant": "C3-S4", "decode_tokens_per_second": 0.6, "ttft_seconds": 1},
                {"tag": "io", "variant": "C3-S4", "decode_tokens_per_second": 0.5, "prefetch_workers": 1, "coalesce_gap": 0},
                {"tag": "length", "variant": "C3-S1", "max_new_tokens": 8, "decode_tokens_per_second": 0.8},
                {"tag": "length", "variant": "C3-S4", "max_new_tokens": 8, "decode_tokens_per_second": 0.7},
            )
        )
        result = recommendations(rows)
        self.assertEqual(result["warm_by_budget"]["1024"]["variant"], "C3-S1")
        self.assertEqual(result["model_cold_4g"]["variant"], "C3-S1")
        self.assertEqual(result["short_output"]["8"]["variant"], "C3-S1")


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
