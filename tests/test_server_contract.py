import http.client
import json
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import patch
from io import StringIO
from contextlib import redirect_stdout

from sparseflow.server import APIError, GenerationScheduler, SparseFlowAPIServer, generation_options, normalize_messages
from sparseflow.serving_types import ContextLengthExceeded, GenerationCancelled, GenerationRequest, GenerationResult, ServingConfig


class ContractEngine:
    model_id = "contract-model"
    preset = "stable"

    def __init__(self):
        self.calls = []
        self.release = threading.Event()
        self.started = threading.Event()
        self.closed = 0

    def snapshot(self):
        return {"state": "ready", "model_id": self.model_id, "preset": self.preset}

    def validate_request(self, request):
        if request.messages[0]["content"] == "too long":
            raise ContextLengthExceeded(7, request.max_completion_tokens, 8)

    def generate(self, request, on_delta=None, is_cancelled=None):
        self.calls.append(request.request_id)
        self.started.set()
        if request.request_id.endswith("slow") or request.stream:
            while not self.release.is_set():
                if is_cancelled and is_cancelled():
                    return GenerationResult("", finish_reason="cancelled")
                time.sleep(0.005)
        if on_delta:
            on_delta("he")
            on_delta("llo")
        return GenerationResult("hello", 2, 2, "stop")

    def close(self):
        self.closed += 1


def config(**kwargs):
    values = {
        "model_dir": Path("E:/models/fake"),
        "int8_container": Path("E:/models/fake-int8"),
        "model_id": "contract-model",
        "port": 0,
        "max_queue": 1,
        "queue_timeout_seconds": 0.2,
    }
    values.update(kwargs)
    return ServingConfig(**values)


class RequestContractTest(unittest.TestCase):
    def test_text_parts_and_developer_role(self):
        messages = normalize_messages([
            {"role": "developer", "content": [{"type": "input_text", "text": "rules"}, {"type": "text", "text": " here"}]},
            {"role": "user", "content": "question"},
        ])
        self.assertEqual(messages[0], {"role": "system", "content": "rules here"})

    def test_multimodal_and_stream_option_types_are_explicit_400(self):
        with self.assertRaises(APIError) as raised:
            normalize_messages([{"role": "user", "content": [{"type": "image_url", "image_url": "x"}]}])
        self.assertEqual((raised.exception.status, raised.exception.code), (400, "unsupported_content_type"))
        for body, param in (
            ({"stream_options": []}, "stream_options"),
            ({"stream_options": {"include_usage": "yes"}}, "stream_options.include_usage"),
        ):
            with self.subTest(param=param):
                with self.assertRaises(APIError) as raised:
                    generation_options(body, 16)
                self.assertEqual(raised.exception.status, 400)
                self.assertEqual(raised.exception.param, param)

    def test_port_zero_and_remote_bind_gate(self):
        self.assertEqual(config().port, 0)
        self.assertEqual(ServingConfig(Path("E:/models/fake"), Path("E:/models/fake-int8")).port, 8000)
        with self.assertRaises(ValueError):
            config(host="0.0.0.0")
        self.assertEqual(config(host="0.0.0.0", api_key="secret").port, 0)

    def test_cli_accepts_port_zero(self):
        from sparseflow.cli import main

        class Httpd:
            server_port = 43210

            def serve_forever(self):
                raise KeyboardInterrupt

        engine = object()
        api = type("Api", (), {"server": lambda self: Httpd(), "close": lambda self: None})()
        output = StringIO()
        with patch("sparseflow.serving.SparseFlowEngine", return_value=engine) as engine_type:
            with patch("sparseflow.server.SparseFlowAPIServer", return_value=api):
                with redirect_stdout(output):
                    self.assertEqual(main([
                        "serve", "E:/models/fake", "--int8-container", "E:/models/fake-int8", "--port", "0",
                    ]), 0)
        self.assertEqual(engine_type.call_args.args[0].port, 0)

    def test_cli_uses_laptop_preset_defaults(self):
        from sparseflow.cli import main

        class Httpd:
            server_port = 43211

            def serve_forever(self):
                raise KeyboardInterrupt

        engine = object()
        api = type("Api", (), {"server": lambda self: Httpd(), "close": lambda self: None})()
        output = StringIO()
        with patch("sparseflow.serving.SparseFlowEngine", return_value=engine) as engine_type:
            with patch("sparseflow.server.SparseFlowAPIServer", return_value=api):
                with redirect_stdout(output):
                    self.assertEqual(main([
                        "serve",
                        "E:/models/fake",
                        "--int8-container",
                        "E:/models/fake-int8",
                        "--preset",
                        "laptop-16gb",
                        "--port",
                        "0",
                    ]), 0)
        config = engine_type.call_args.args[0]
        self.assertEqual(config.preset, "laptop-16gb")
        self.assertEqual(config.ctx, 2048)
        self.assertEqual(config.context_tokens, 2048)
        self.assertEqual(config.max_completion_tokens, 128)


class SchedulerContractTest(unittest.TestCase):
    def test_timeout_and_queued_cancel_remove_tickets(self):
        engine = ContractEngine()
        scheduler = GenerationScheduler(engine, max_queue=1, queue_timeout_seconds=0.03)
        first = threading.Thread(
            target=lambda: scheduler.submit(GenerationRequest("req_slow", "contract-model", ({"role": "user", "content": "x"},), 2))
        )
        first.start()
        self.assertTrue(engine.started.wait(1))
        timeout_result = []

        def wait_timeout():
            try:
                scheduler.submit(GenerationRequest("req_timeout", "contract-model", ({"role": "user", "content": "x"},), 2))
            except APIError as exc:
                timeout_result.append(exc.code)

        waiter = threading.Thread(target=wait_timeout)
        waiter.start()
        waiter.join(1)
        self.assertEqual(timeout_result, ["queue_timeout"])

        queued_result = []

        def submit_queued():
            queued_result.append(scheduler.submit(GenerationRequest("req_queued", "contract-model", ({"role": "user", "content": "x"},), 2)))

        queued = threading.Thread(target=submit_queued)
        queued.start()
        time.sleep(0.01)
        self.assertEqual(scheduler.cancel("req_queued"), "cancellation_requested")
        queued.join(1)
        self.assertEqual(engine.calls, ["req_slow"])
        engine.release.set()
        first.join(1)
        deadline = time.monotonic() + 1
        while scheduler.snapshot()["active"] and time.monotonic() < deadline:
            time.sleep(0.005)
        with scheduler._condition:
            self.assertEqual(scheduler._tickets, {})
        scheduler.close()

    def test_close_wakes_waiters_and_cancel_lifecycle(self):
        engine = ContractEngine()
        scheduler = GenerationScheduler(engine, max_queue=1, queue_timeout_seconds=5)
        active = threading.Thread(
            target=lambda: scheduler.submit(GenerationRequest("req_slow", "contract-model", ({"role": "user", "content": "x"},), 2))
        )
        active.start()
        self.assertTrue(engine.started.wait(1))
        waiting_errors = []

        def submit_waiting():
            try:
                scheduler.submit(GenerationRequest("req_wait", "contract-model", ({"role": "user", "content": "x"},), 2))
            except APIError as exc:
                waiting_errors.append(exc.code)

        waiting = threading.Thread(
            target=submit_waiting
        )
        waiting.start()
        time.sleep(0.01)
        scheduler.close()
        waiting.join(1)
        active.join(1)
        self.assertFalse(waiting.is_alive())
        self.assertEqual(waiting_errors, ["scheduler_closed"])
        self.assertEqual(scheduler.snapshot()["queued"], 0)
        with self.assertRaises(APIError) as finished:
            scheduler.cancel("req_slow")
        self.assertEqual(finished.exception.code, "generation_finished")
        with self.assertRaises(APIError) as unknown:
            scheduler.cancel("missing")
        self.assertEqual(unknown.exception.code, "generation_not_found")

    def test_close_releases_queued_submit_with_scheduler_closed(self):
        engine = ContractEngine()
        scheduler = GenerationScheduler(engine, max_queue=1, queue_timeout_seconds=5)
        active = threading.Thread(
            target=lambda: scheduler.submit(GenerationRequest("req_slow", "contract-model", ({"role": "user", "content": "x"},), 2))
        )
        active.start()
        self.assertTrue(engine.started.wait(1))
        errors = []

        def wait_for_admission():
            try:
                scheduler.submit(GenerationRequest("req_wait", "contract-model", ({"role": "user", "content": "x"},), 2))
            except APIError as exc:
                errors.append(exc.code)

        waiting = threading.Thread(target=wait_for_admission)
        waiting.start()
        time.sleep(0.01)
        scheduler.close()
        waiting.join(1)
        active.join(1)
        self.assertEqual(errors, ["scheduler_closed"])

    def test_generation_cancelled_exception_is_a_cancelled_sse_finish(self):
        class RaisingEngine(ContractEngine):
            def generate(self, request, on_delta=None, is_cancelled=None):
                raise GenerationCancelled("cancelled")

        engine = RaisingEngine()
        scheduler = GenerationScheduler(engine, max_queue=1, queue_timeout_seconds=1)
        ticket = scheduler.submit(GenerationRequest("req_cancel", "contract-model", ({"role": "user", "content": "x"},), 2, stream=True))
        self.assertTrue(ticket.done.wait(1))
        self.assertEqual(ticket.result.finish_reason, "cancelled")
        self.assertEqual(ticket.terminal_status, "cancelled")
        scheduler.close()


class SocketContractTest(unittest.TestCase):
    def setUp(self):
        self.engine = ContractEngine()
        self.api = SparseFlowAPIServer(self.engine, config())
        self.httpd = self.api.server()
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.engine.release.set()
        self.api.close()
        self.thread.join(1)

    def post(self, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=2)
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        response = connection.getresponse()
        payload = response.read()
        headers = dict(response.getheaders())
        connection.close()
        return response.status, headers, payload

    def test_context_validation_is_a_400(self):
        status, _, payload = self.post({"model": "contract-model", "messages": [{"role": "user", "content": "too long"}]})
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(payload)["error"]["code"], "context_length_exceeded")

    def test_sse_and_active_cancel_use_real_socket(self):
        connection = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=2)
        connection.request("POST", "/v1/chat/completions", json.dumps({
            "model": "contract-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        }), {"Content-Type": "application/json"})
        response = connection.getresponse()
        self.assertEqual(response.status, 200)
        request_id = response.getheader("x-request-id")
        cancel = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=2)
        cancel.request("POST", f"/v1/generations/{request_id}/cancel", "{}", {"Content-Type": "application/json"})
        cancel_response = cancel.getresponse()
        self.assertEqual(cancel_response.status, 202)
        cancel.close()
        data = response.read().decode("utf-8")
        connection.close()
        self.assertIn("data:", data)
        self.assertIn("[DONE]", data)
        self.assertEqual(self.api.scheduler.snapshot()["queued"], 0)
