import unittest

from benchmarks.run_stage7_11_cache_pilot import _parse_bytes
from benchmarks.run_stage7_11_laptop import compact_result
from benchmarks.summarize_stage7_11_laptop import summarize
from benchmarks.verify_stage7_11_laptop import verify


def fake_result():
    return {
        "generated_ids": [1, 2],
        "text": "ok",
        "generated_tokens": 2,
        "logit_fingerprints": [{"sha256": "logit"}],
        "route_audit": [{"layer": 0, "expert_ids": [1, 2], "sha256": "route"}],
        "loader": {"load_seconds": 1.5},
        "prefill_seconds": 2.0,
        "decode_seconds": 1.0,
        "decode_token_seconds": [1.0],
        "runtime_identity": {"runtime_id": "test"},
        "memory": {
            "rss_after_generation": 100,
            "process_peak_rss": 120,
            "private_bytes_after_generation": 140,
            "read_bytes": 160,
            "read_bytes_semantics": "windows-process-read-transfer",
        },
        "cache": {"hits": 1, "misses": 2, "evictions": 0, "cached_bytes": 64, "pinned_entries": 0},
        "provider_storage": {"reader_bytes": 200, "transient_prefetch_entries": 0},
    }


class Stage711LaptopTest(unittest.TestCase):
    def test_cache_pilot_parser_uses_binary_units(self):
        self.assertEqual(_parse_bytes("128MiB"), 128 * 1024**2)
        self.assertEqual(_parse_bytes("4GiB"), 4 * 1024**3)

    def test_summary_keeps_token_lengths_as_separate_cells(self):
        base = compact_result(
            fake_result(),
            prompt_id="en-explain",
            category="en",
            repeat=1,
            max_new_tokens=8,
            cache_bytes=1024,
            context_tokens=2048,
            wall_seconds=3.0,
        )
        second = dict(base, repeat=2)
        long = dict(base, repeat=1, max_new_tokens=16, generated_ids_hash="long")
        long_second = dict(long, repeat=2)
        artifact = {
            "kind": "sparseflow_stage7_11_laptop_cli",
            "protocol": {"prompt_count": 1, "repeats": 2, "token_counts": [8, 16]},
            "samples": [
                dict(base, exit_code=0),
                dict(second, exit_code=0),
                dict(long, exit_code=0),
                dict(long_second, exit_code=0),
            ],
        }
        summary = summarize(artifact)
        self.assertEqual([(cell["prompt_id"], cell["max_new_tokens"]) for cell in summary["cells"]], [("en-explain", 8), ("en-explain", 16)])
        self.assertFalse(summary["gates"]["passed"])
        self.assertFalse(summary["gates"]["formal_matrix_shape"])
        self.assertFalse(summary["gates"]["performance_threshold_configured"])

    def test_compact_result_removes_large_route_payload(self):
        result = compact_result(
            fake_result(),
            prompt_id="en-explain",
            category="en",
            repeat=1,
            max_new_tokens=2,
            cache_bytes=1024,
            context_tokens=2048,
            wall_seconds=3.0,
        )
        self.assertNotIn("route_audit", result)
        self.assertNotIn("captured_logits", result)
        self.assertTrue(result["route_fingerprint"])
        self.assertEqual(result["leases_after"], 0)

    def test_verifier_accepts_clean_exact_compact_fixture(self):
        left = compact_result(
            fake_result(),
            prompt_id="en-explain",
            category="en",
            repeat=1,
            max_new_tokens=2,
            cache_bytes=1024,
            context_tokens=2048,
            wall_seconds=3.0,
        )
        right = dict(left, repeat=2)
        artifact = {
            "kind": "sparseflow_stage7_11_laptop_cli",
            "agent": "Benchmark",
            "protocol": {"mode": "process-cold"},
            "git": {"commit": "abc", "dirty": False},
            "model": {"metadata_sha256": "model"},
            "container": {"metadata_sha256": "container"},
            "samples": [dict(left, exit_code=0), dict(right, exit_code=0)],
        }
        verification = verify(artifact)
        self.assertTrue(verification["verification_passed"], verification)

    def test_verifier_compares_repeats_per_token_count(self):
        left = compact_result(
            fake_result(),
            prompt_id="en-explain",
            category="en",
            repeat=1,
            max_new_tokens=8,
            cache_bytes=1024,
            context_tokens=2048,
            wall_seconds=3.0,
        )
        right = dict(left, repeat=2)
        longer = dict(left, repeat=1, max_new_tokens=16, generated_ids_hash="longer")
        longer_repeat = dict(longer, repeat=2)
        artifact = {
            "kind": "sparseflow_stage7_11_laptop_cli",
            "agent": "Benchmark",
            "protocol": {
                "mode": "process-cold",
                "prompt_count": 1,
                "repeats": 2,
                "token_counts": [8, 16],
            },
            "git": {"commit": "abc", "dirty": False},
            "model": {"metadata_sha256": "model"},
            "container": {"metadata_sha256": "container"},
            "samples": [
                dict(left, exit_code=0),
                dict(right, exit_code=0),
                dict(longer, exit_code=0),
                dict(longer_repeat, exit_code=0),
            ],
        }
        verification = verify(artifact)
        self.assertFalse(verification["verification_passed"], verification)
        self.assertIn("formal_matrix_shape", verification["failures"])

    def test_verifier_rejects_dirty_formal_fixture(self):
        artifact = {
            "kind": "sparseflow_stage7_11_laptop_cli",
            "agent": "Benchmark",
            "protocol": {"mode": "process-cold"},
            "git": {"commit": "abc", "dirty": True},
            "model": {"metadata_sha256": "model"},
            "container": {"metadata_sha256": "container"},
            "samples": [],
        }
        verification = verify(artifact)
        self.assertFalse(verification["verification_passed"])
        self.assertIn("clean_commit", verification["failures"])


if __name__ == "__main__":
    unittest.main()
