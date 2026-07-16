import unittest

from benchmarks.freeze_quality_manifest import normalize_row


class QualityManifestTest(unittest.TestCase):
    def test_normalizes_all_standard_tasks(self):
        hella = normalize_row(
            "hellaswag",
            2,
            {"ctx": "A person", "endings": ["runs", "sleeps"], "label": "1", "ind": 7},
        )
        self.assertEqual(hella["gold"], 1)
        self.assertEqual(hella["choices"], [" runs", " sleeps"])

        arc = normalize_row(
            "arc_challenge",
            3,
            {
                "id": "arc-3",
                "question": "Why?",
                "choices": {"label": ["A", "B"], "text": ["One", "Two"]},
                "answerKey": "B",
            },
        )
        self.assertEqual(arc["gold"], 1)
        self.assertTrue(arc["ctx"].endswith("Answer:"))

        mmlu = normalize_row(
            "mmlu",
            4,
            {"question": "Pick", "choices": ["A", "B"], "answer": 0, "subject": "math"},
        )
        self.assertEqual(mmlu["source_id"], "math:4")
        self.assertEqual(mmlu["gold"], 0)


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
