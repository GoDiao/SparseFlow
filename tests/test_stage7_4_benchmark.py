import tempfile
import unittest
from pathlib import Path

from benchmarks.common import numeric_delta, parse_bytes, percentile
from benchmarks.prepare_generic_offload import expected_dat_bytes
from benchmarks.run_sparseflow import validate_cache_state
from benchmarks.run_stage7_4_matrix import core_matrix
from benchmarks.summarize_stage7_4 import aggregate, validate_results


class Stage74CommonTest(unittest.TestCase):
    def test_byte_parser_and_percentile(self):
        self.assertEqual(parse_bytes("4GiB"), 4 * 1024**3)
        self.assertEqual(parse_bytes("512 MiB"), 512 * 1024**2)
        self.assertEqual(percentile([1.0, 3.0], 0.5), 2.0)
        with self.assertRaises(ValueError):
            parse_bytes("broken")

    def test_nested_numeric_delta_preserves_metadata_and_booleans(self):
        self.assertEqual(
            numeric_delta(
                {"count": 2, "nested": {"seconds": 1.5}, "ok": False},
                {"count": 7, "nested": {"seconds": 2.0}, "ok": True},
            ),
            {"count": 5, "nested": {"seconds": 0.5}, "ok": True},
        )

    def test_cache_state_contract(self):
        validate_cache_state("workload-warm", warmup=1, runs=3)
        validate_cache_state("model-cold", warmup=0, runs=1)
        with self.assertRaises(ValueError):
            validate_cache_state("workload-warm", warmup=0, runs=3)
        with self.assertRaises(ValueError):
            validate_cache_state("model-cold", warmup=0, runs=3)

    def test_core_matrix_is_frozen(self):
        cells = core_matrix()
        self.assertEqual(len(cells), 27)
        self.assertEqual(sum(cell.cache_state == "model-cold" for cell in cells), 9)
        self.assertEqual(
            {(cell.variant, cell.cache_bytes // 1024**3) for cell in cells if cell.variant == "C3-S4" and cell.cache_state == "workload-warm"},
            {("C3-S4", 1), ("C3-S4", 2), ("C3-S4", 4), ("C3-S4", 8)},
        )

    def test_generic_offload_size(self):
        self.assertEqual(expected_dat_bytes("BF16", [256, 1024]), 512 * 1024)


def _benchmark_result(variant: str, cache_bytes: int | None, reader_bytes: int):
    streaming = variant != "C3-R"
    provider = {
        "reader_bytes": reader_bytes,
        "cache_hits": 2 if streaming else 0,
        "cache_misses": 1 if streaming else 0,
        "demand_requests": 3 if streaming else 0,
        "demand_reuse_hits": 2 if streaming else 0,
        "demand_prefetch_served": 0,
        "demand_misses": 1 if streaming else 0,
        "prefetch_wasted_ready_bytes": 0,
    }
    fingerprints = [{"sha256": "same", "shape": [1, 3], "dtype": "bfloat16"}]
    return {
        "kind": "sparseflow_stage7_4_benchmark",
        "stage": "7.4",
        "agent": "Main Dev",
        "variant": variant,
        "model": {"config_sha256": "config", "index_sha256": "index"},
        "git": {"commit": "abc", "dirty": False},
        "storage_policy": {"cache_bytes": cache_bytes, "cache_state": "workload-warm"},
        "load": {
            "seconds": 1.0,
            "loader": {
                "expert_reader_calls_after_init": 0,
                "expert_reader_bytes_after_init": 0,
            },
        },
        "runs": [
            {
                "runtime_identity": {"kernel_id": "same"},
                "quality": {
                    "generated_ids": [1, 2],
                    "logit_fingerprints": fingerprints,
                },
                "timing": {
                    "time_to_first_token_seconds": 2.0,
                    "decode_tokens_per_second": 0.5,
                    "decode_token_seconds": [2.0],
                    "decode_tokens": 1,
                },
                "memory": {"process_peak_rss": 10},
                "process_metrics_delta": {"read_bytes": reader_bytes},
                "provider_delta": provider,
                "cache_after": {"cached_bytes": min(cache_bytes or 0, 6)},
                "prefetch_after": {"failed": 0},
                "telemetry": {
                    "forwards": [
                        {
                            "phase": "decode",
                            "provider": {"reader_bytes": reader_bytes},
                        }
                    ]
                },
            }
        ],
        "summary": {},
        "_path": f"{variant}.json",
    }


class Stage74SummaryTest(unittest.TestCase):
    def test_validation_and_aggregation(self):
        results = [
            _benchmark_result("C3-R", None, 0),
            _benchmark_result("C3-S1", 1024, 600),
        ]
        validation = validate_results(results)
        self.assertTrue(validation["all_invariants_pass"])
        rows = aggregate(results)
        streaming = next(row for row in rows if row["variant"] == "C3-S1")
        self.assertEqual(streaming["median_decode_reader_bytes_per_token"], 600)
        self.assertAlmostEqual(streaming["cache_hit_rate"], 2 / 3)


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
