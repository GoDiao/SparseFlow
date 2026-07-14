import unittest
from types import SimpleNamespace

try:
    import torch
except ImportError:  # pragma: no cover - benchmark host normally has torch
    torch = None

from benchmarks.run_cpu import greedy_generate_timed


@unittest.skipIf(torch is None, "PyTorch is required for benchmark timing tests")
class BenchmarkTimingTest(unittest.TestCase):
    def test_decode_excludes_first_prefill_token(self):
        class TinyModel:
            def __call__(
                self,
                input_ids,
                attention_mask,
                use_cache,
                past_key_values=None,
            ):
                logits = torch.zeros((1, input_ids.shape[1], 5))
                logits[:, -1, 1] = 1
                return SimpleNamespace(logits=logits, past_key_values=object())

        result = greedy_generate_timed(
            TinyModel(),
            {
                "input_ids": torch.tensor([[2, 3]]),
                "attention_mask": torch.ones((1, 2), dtype=torch.long),
            },
            max_new_tokens=3,
            torch=torch,
        )

        self.assertEqual(tuple(result["output"].shape), (1, 5))
        self.assertEqual(result["decode_tokens"], 2)
        self.assertEqual(len(result["decode_token_seconds"]), 2)
        self.assertGreaterEqual(result["end_to_end_seconds"], result["prefill_seconds"])
        self.assertGreater(result["decode_seconds"], 0)


if __name__ == "__main__":
    unittest.main()


# [Benchmark]
