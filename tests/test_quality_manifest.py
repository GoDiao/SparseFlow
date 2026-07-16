import unittest

import torch

from benchmarks.freeze_quality_manifest import normalize_row
from benchmarks.score_choices import score_batch, score_one


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

    def test_batched_choice_scoring_matches_sequential_scoring(self):
        class Tokenizer:
            pad_token_id = 0
            eos_token_id = 0

            def __call__(self, text, add_special_tokens=False):
                del add_special_tokens
                return {"input_ids": [ord(value) % 11 + 1 for value in text]}

        class Model:
            def __call__(self, input_ids, attention_mask, use_cache=False):
                del attention_mask, use_cache
                batch, length = input_ids.shape
                logits = torch.arange(16, dtype=torch.float32).view(1, 1, 16)
                return type("Output", (), {"logits": logits.expand(batch, length, 16)})()

        tokenizer = Tokenizer()
        model = Model()
        choices = [" x", " longer"]
        sequential = [score_one(model, tokenizer, "ctx", choice, torch) for choice in choices]
        batched, _elapsed = score_batch(model, tokenizer, "ctx", choices, torch)
        for left, right in zip(sequential, batched):
            self.assertAlmostEqual(left["loglikelihood"], right["loglikelihood"], places=6)
            self.assertEqual(left["token_loglikelihoods"], right["token_loglikelihoods"])


if __name__ == "__main__":
    unittest.main()


# [Main Dev]
