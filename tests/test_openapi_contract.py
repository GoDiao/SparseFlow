import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
OPENAPI = ROOT / "docs" / "openapi" / "sparseflow-openapi.json"
EXAMPLES = ROOT / "docs" / "api_examples"


class OpenAPIContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads(OPENAPI.read_text(encoding="utf-8"))

    def test_spec_and_frontend_paths_are_present(self):
        self.assertEqual(self.spec["openapi"], "3.1.0")
        expected = {
            "/health",
            "/v1/models",
            "/v1/models/{model_id}",
            "/v1/runtime",
            "/v1/chat/completions",
            "/v1/generations/{request_id}/cancel",
        }
        self.assertEqual(set(self.spec["paths"]), expected)
        self.assertEqual(self.spec["paths"]["/health"]["get"]["security"], [])

    def test_runtime_and_health_examples_have_required_fields(self):
        health_required = {"status", "ready", "model", "preset", "scheduler", "runtime"}
        for name in ("health_loading.json", "health_ready.json"):
            value = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
            self.assertTrue(health_required <= value.keys())
            self.assertIn(value["status"], {"loading", "ready"})
            self.assertFalse(value["runtime"]["persistent_kv_supported"])

        runtime = json.loads((EXAMPLES / "runtime_ready.json").read_text(encoding="utf-8"))
        required = {
            "schema_version", "state", "model_id", "preset", "public_status",
            "effective_config", "runtime_load_count", "runtime_load_seconds",
            "runtime_identity", "model", "container", "process_memory", "last_generation",
        }
        self.assertTrue(required <= runtime.keys())
        self.assertEqual(runtime["runtime_load_count"], 1)
        self.assertEqual(runtime["evidence"], "SIMULATED")
        self.assertEqual(runtime["effective_config"]["context_tokens"], 2048)
        self.assertEqual(runtime["process_memory"]["peak_rss_bytes"], 6653952000)

    def test_completion_and_errors_match_frozen_shapes(self):
        completion = json.loads((EXAMPLES / "chat_non_stream.json").read_text(encoding="utf-8"))
        self.assertEqual(completion["object"], "chat.completion")
        self.assertEqual(completion["choices"][0]["message"]["role"], "assistant")
        self.assertEqual(set(completion["usage"]), {"prompt_tokens", "completion_tokens", "total_tokens"})

        for name in ("error_memory_admission.json", "error_context_length.json"):
            error = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))["error"]
            self.assertEqual(set(error), {"message", "type", "param", "code"})

    def test_sse_fixture_has_keepalive_chunks_and_done(self):
        raw = (EXAMPLES / "chat_stream.sse").read_text(encoding="utf-8")
        self.assertIn(": sparseflow-keepalive", raw)
        data_lines = [line[6:] for line in raw.splitlines() if line.startswith("data: ")]
        self.assertEqual(data_lines[-1], "[DONE]")
        chunks = [json.loads(line) for line in data_lines[:-1]]
        self.assertTrue(all(chunk["object"] == "chat.completion.chunk" for chunk in chunks))
        self.assertEqual(chunks[-1]["choices"], [])

    def test_openapi_does_not_advertise_unimplemented_capabilities(self):
        request_properties = self.spec["components"]["schemas"]["ChatCompletionRequest"]["properties"]
        for unsupported in ("tools", "functions", "tool_choice", "images", "vision", "cache_slot"):
            self.assertNotIn(unsupported, request_properties)
        session = self.spec["components"]["schemas"]["RuntimeSession"]
        self.assertEqual(session["properties"]["persistent_kv_supported"]["const"], False)


if __name__ == "__main__":
    unittest.main()
