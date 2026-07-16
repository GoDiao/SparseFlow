import unittest

from benchmarks.summarize_stage7_5_formal import quality_metrics, wilson


class Stage75FormalSummaryTest(unittest.TestCase):
    def test_wilson_and_task_quality_metrics(self):
        interval = wilson(10, 20)
        self.assertLess(interval[0], 0.5)
        self.assertGreater(interval[1], 0.5)
        rows = []
        for task in ("hellaswag", "arc_challenge", "mmlu"):
            rows.append(
                {
                    "task": task,
                    "correct": True,
                    "correct_norm_char": True,
                    "correct_norm_token": False,
                }
            )
        result = quality_metrics(rows)
        self.assertEqual(result["aggregate"]["accuracy"], 1.0)
        self.assertEqual(result["aggregate"]["acc_norm_token"], 0.0)
        self.assertEqual(result["tasks"]["mmlu"]["n"], 1)


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
