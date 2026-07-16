import unittest

from benchmarks.native_stage7_5 import extract_record, paired_order, summarize


def report(speed: float):
    path = {
        "generated_tokens": 3,
        "decode_seconds": 2.0 / speed,
        "load_seconds": 1.0,
        "prefill_seconds": 2.0,
        "memory": {"rss_after_generation": 10},
        "generation_expert_io": {"read_bytes": 0},
        "provider_storage": {},
        "generated_ids": [1, 2, 3],
        "logit_fingerprints": [{"sha256": "a"}],
    }
    return {
        "all_invariants_pass": True,
        "runtime_identity": {},
        "correctness": {"all_equal": True},
        "invariants": {},
        "resident": dict(path),
        "streaming": dict(path),
    }


class NativeStage75BenchmarkTest(unittest.TestCase):
    def test_alternates_order_and_reports_speedup(self):
        self.assertEqual(paired_order(0), ("reference", "native"))
        self.assertEqual(paired_order(1), ("native", "reference"))
        records = []
        for repetition in range(3):
            records.append(extract_record("reference", repetition, report(1.0)))
            records.append(extract_record("native", repetition, report(1.25)))
        summary = summarize(records)
        self.assertEqual(summary["reference"]["runs"], 3)
        self.assertAlmostEqual(summary["native_resident_speedup"], 1.25)
        self.assertAlmostEqual(summary["native_streaming_speedup"], 1.25)


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
