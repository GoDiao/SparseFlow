import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import torch
from torch import nn

from sparseflow.cli import main
from sparseflow.text_runtime import (
    Qwen36TextRuntime,
    compare_generation_results,
    compare_text_paths,
)


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, *_args, **_kwargs):
        return {
            "input_ids": torch.tensor([[10, 11]], dtype=torch.long),
            "attention_mask": torch.ones(1, 2, dtype=torch.long),
        }

    def decode(self, token_ids, **_kwargs):
        return "tokens:" + ",".join(str(token) for token in token_ids)


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.calls = []

    def forward(self, input_ids, attention_mask, past_key_values=None, use_cache=True, **_kwargs):
        self.calls.append(
            {
                "input_shape": tuple(input_ids.shape),
                "mask_shape": tuple(attention_mask.shape),
                "has_past": past_key_values is not None,
                "use_cache": use_cache,
            }
        )
        vocab = 16
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], vocab)
        token = 2 + len(self.calls) - 1
        logits[:, -1, token] = 1.0
        return types.SimpleNamespace(logits=logits, past_key_values=object())


class TextRuntimeTest(unittest.TestCase):
    def test_prefill_decode_and_greedy_generation(self):
        model = FakeModel()
        runtime = Qwen36TextRuntime(
            model,
            FakeTokenizer(),
            torch,
            "/tmp/fake-qwen36",
            mode="resident",
        )
        result = runtime.greedy_generate(
            "hello",
            max_new_tokens=3,
            stop_on_eos=True,
            record_logit_fingerprints=True,
        )

        self.assertEqual(result["generated_ids"], [2, 3, 4])
        self.assertEqual(result["generated_tokens"], 3)
        self.assertEqual(result["text"], "tokens:2,3,4")
        self.assertGreaterEqual(result["prefill_seconds"], 0.0)
        self.assertEqual(len(result["decode_token_seconds"]), 2)
        self.assertEqual(len(result["logit_fingerprints"]), 3)
        self.assertEqual(
            [item["argmax"] for item in result["logit_fingerprints"]],
            [[2], [3], [4]],
        )
        self.assertEqual([call["has_past"] for call in model.calls], [False, True, True])
        self.assertEqual([call["input_shape"] for call in model.calls], [(1, 2), (1, 1), (1, 1)])
        self.assertTrue(all(call["use_cache"] for call in model.calls))

    def test_generation_comparison_requires_complete_external_equality(self):
        resident = {
            "input_ids": [1, 2],
            "generated_ids": [3, 4],
            "generated_tokens": 2,
            "text": "ok",
            "logit_fingerprints": [{"sha256": "same"}],
        }
        self.assertTrue(compare_generation_results(resident, dict(resident))["all_equal"])

        changed = dict(resident)
        changed["generated_ids"] = [3, 5]
        comparison = compare_generation_results(resident, changed)
        self.assertFalse(comparison["generated_ids_equal"])
        self.assertFalse(comparison["all_equal"])

        changed = dict(resident)
        changed["logit_fingerprints"] = [{"sha256": "different"}]
        comparison = compare_generation_results(resident, changed)
        self.assertFalse(comparison["logit_fingerprints_equal"])
        self.assertFalse(comparison["all_equal"])

    def test_text_check_cli_emits_comparison_and_uses_failure_exit_code(self):
        result = {
            "correctness": {"all_equal": True},
            "resident": {
                "generated_ids": [3],
                "text": "ok",
                "load_seconds": 1.0,
                "prefill_seconds": 2.0,
                "decode_seconds": 0.0,
            },
            "streaming": {
                "generated_ids": [3],
                "text": "ok",
                "load_seconds": 1.5,
                "prefill_seconds": 3.0,
                "decode_seconds": 0.0,
                "cache": {"requests": 2, "hits": 1, "misses": 1, "evictions": 0},
            },
        }
        output = StringIO()
        with patch("sparseflow.cli.compare_text_paths", return_value=result):
            with redirect_stdout(output):
                exit_code = main(
                    [
                        "text-check",
                        "/tmp/model",
                        "--prompt",
                        "hello",
                    ]
                )
        self.assertEqual(exit_code, 0)
        self.assertIn("SparseFlow text-check", output.getvalue())

        result["correctness"]["all_equal"] = False
        with patch("sparseflow.cli.compare_text_paths", return_value=result):
            with redirect_stdout(StringIO()):
                exit_code = main(
                    [
                        "text-check",
                        "/tmp/model",
                        "--prompt",
                        "hello",
                    ]
                )
        self.assertEqual(exit_code, 1)

    def test_text_check_pins_eager_arithmetic_for_both_paths(self):
        result = {
            "input_ids": [1],
            "generated_ids": [2],
            "generated_tokens": 1,
            "text": "ok",
            "logit_fingerprints": [{"sha256": "same"}],
        }
        with patch(
            "sparseflow.text_runtime._run_text_path",
            side_effect=[(dict(result), 1.0), (dict(result), 2.0)],
        ) as run_path:
            comparison = compare_text_paths(
                "/tmp/model",
                "hello",
                max_new_tokens=1,
            )

        self.assertTrue(comparison["correctness"]["all_equal"])
        self.assertEqual(comparison["experts_implementation"], "eager")
        self.assertEqual(run_path.call_count, 2)
        self.assertEqual(run_path.call_args_list[0].kwargs["mode"], "resident")
        self.assertEqual(run_path.call_args_list[1].kwargs["mode"], "streaming")
        self.assertEqual(
            [call.kwargs["experts_implementation"] for call in run_path.call_args_list],
            ["eager", "eager"],
        )


if __name__ == "__main__":
    unittest.main()
