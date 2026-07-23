import http.client
import json
from pathlib import Path
import socket
import threading
import time
import unittest

from sparseflow.server import GenerationScheduler, SparseFlowAPIServer
from sparseflow.serving_types import GenerationRequest, GenerationResult, ServingConfig


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeEngine:
    model_id = "fake-model"
    preset = "stable"

    def __init__(self, slow=False):
        self.calls = []
        self.closed = 0
        self.slow = slow

    def snapshot(self):
        return {"state": "ready", "model_id": self.model_id, "preset": self.preset}

    def generate(self, request, on_delta=None, is_cancelled=None):
        self.calls.append(request.request_id)
        for part in ("he", "llo"):
            if self.slow:
                time.sleep(0.03)
            if is_cancelled and is_cancelled():
                return GenerationResult("", finish_reason="cancelled")
            if on_delta:
                on_delta(part)
        return GenerationResult("hello", 3, 2, "stop", telemetry={"fake": True})

    def cancel(self, request_id):
        raise RuntimeError("not active")

    def close(self):
        self.closed += 1


class ServerTest(unittest.TestCase):
    def setUp(self):
        self.engine = FakeEngine()
        config = ServingConfig(Path("E:/models/fake"), Path("E:/models/fake-int8"), model_id="fake-model", port=free_port(), max_queue=2)
        self.api = SparseFlowAPIServer(self.engine, config)
        self.httpd = self.api.server()
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.api.close()
        self.thread.join(timeout=2)

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=3)
        request_headers = {"Content-Type": "application/json", **(headers or {})}
        connection.request(method, path, body=json.dumps(body) if body is not None else None, headers=request_headers)
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, dict(response.getheaders()), data

    def body(self, status, headers, data):
        self.assertEqual(status, 200)
        return json.loads(data)

    def test_health_models_and_runtime(self):
        status, _, data = self.request("GET", "/health")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(data)["ready"])
        status, _, data = self.request("GET", "/v1/models")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["data"][0]["id"], "fake-model")
        self.assertEqual(self.request("GET", "/v1/runtime")[0], 200)

    def test_non_streaming_chat(self):
        status, headers, data = self.request("POST", "/v1/chat/completions", {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 4})
        value = self.body(status, headers, data)
        self.assertEqual(value["choices"][0]["message"]["content"], "hello")
        self.assertEqual(value["usage"]["completion_tokens"], 2)
        self.assertIn("x-request-id", {key.lower(): value for key, value in headers.items()})

    def test_streaming_chat_and_unsupported_sampling(self):
        status, headers, data = self.request("POST", "/v1/chat/completions", {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "max_completion_tokens": 4, "stream": True, "stream_options": {"include_usage": True}})
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers["Content-Type"])
        self.assertIn("[DONE]", data.decode("utf-8"))
        status, _, data = self.request("POST", "/v1/chat/completions", {"model": "fake-model", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.5})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data)["error"]["code"], "sampling_not_supported")

    def test_bearer_auth_and_cors(self):
        object.__setattr__(self.api.config, "api_key", "secret")
        self.assertEqual(self.request("GET", "/v1/models")[0], 401)
        status, headers, _ = self.request("GET", "/v1/models", headers={"Authorization": "Bearer secret", "Origin": "http://127.0.0.1:5173"})
        self.assertEqual(status, 200)
        self.assertEqual(headers["Access-Control-Allow-Origin"], "http://127.0.0.1:5173")


class SchedulerTest(unittest.TestCase):
    def test_scheduler_calls_engine_once_and_closes(self):
        engine = FakeEngine()
        scheduler = GenerationScheduler(engine, max_queue=1, queue_timeout_seconds=2)
        request = GenerationRequest("req_1", "fake-model", ({"role": "user", "content": "hi"},), 4)
        ticket = scheduler.submit(request)
        self.assertTrue(ticket.done.wait(2))
        self.assertEqual(ticket.result.text, "hello")
        scheduler.close()
        engine.close()
        self.assertEqual(engine.calls, ["req_1"])
