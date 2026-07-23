import unittest
from pathlib import Path
import threading
import time

from sparseflow.serving import SparseFlowEngine
from sparseflow.serving_types import GenerationCancelled, GenerationRequest, RuntimeState, ServingConfig


class FakeRuntime:
    runtime_identity = {"runtime_id": "fake"}

    def __init__(self):
        self.mode = "success"
        self.closed = 0
        self.started = threading.Event()

    def generate_messages(self, messages, **kwargs):
        del messages
        self.started.set()
        if self.mode == "error":
            raise RuntimeError("boom")
        if self.mode == "cancel":
            while not kwargs["is_cancelled"]():
                time.sleep(0.005)
            raise GenerationCancelled("cancelled")
        if kwargs.get("on_text_delta"):
            kwargs["on_text_delta"]("ok")
        return {"text": "ok", "input_ids": [1, 2], "generated_ids": [3], "generated_tokens": 1}

    def close(self):
        self.closed += 1


class ServingContractTest(unittest.TestCase):
    def test_only_public_presets_are_allowed(self):
        with self.assertRaises(ValueError):
            ServingConfig(Path("E:/model"), Path("E:/container"), preset="experimental-batch")
        laptop = ServingConfig(
            Path("E:/model"),
            Path("E:/container"),
            preset="laptop-16gb",
            ctx=2048,
            context_tokens=2048,
            max_completion_tokens=128,
        )
        self.assertEqual(laptop.preset, "laptop-16gb")
        self.assertEqual(laptop.context_tokens, 2048)
        self.assertEqual(laptop.max_completion_tokens, 128)

    def test_snapshot_is_dependency_free_before_loading(self):
        engine = SparseFlowEngine(ServingConfig(Path("E:/model"), Path("E:/container")), load_async=False)
        snapshot = engine.snapshot()
        self.assertEqual(snapshot["state"], "loading")
        self.assertFalse(snapshot["persistent_kv_supported"])
        engine.close()
        self.assertEqual(engine.snapshot()["state"], "stopped")

    def test_success_exception_and_cancel_restore_ready(self):
        engine = SparseFlowEngine(ServingConfig(Path("E:/model"), Path("E:/container")), load_async=False)
        runtime = FakeRuntime()
        with engine._lock:
            engine._runtime = runtime
            engine._state = RuntimeState.READY

        request = GenerationRequest(
            "req", "qwen3.6-35b-a3b-sparseflow", ({"role": "user", "content": "hi"},), 2
        )
        self.assertEqual(engine.generate(request).text, "ok")
        self.assertEqual(engine.snapshot()["state"], "ready")
        self.assertEqual(engine.snapshot()["generation_count"], 1)
        self.assertEqual(engine.snapshot()["last_request_id"], "req")
        self.assertEqual(engine.snapshot()["last_generation"]["request_id"], "req")
        self.assertEqual(engine.snapshot()["effective_config"]["context_tokens"], 4096)

        runtime.mode = "error"
        with self.assertRaises(RuntimeError):
            engine.generate(request)
        self.assertEqual(engine.snapshot()["state"], "ready")

        runtime.mode = "cancel"
        runtime.started.clear()
        result = []
        worker = threading.Thread(target=lambda: result.append(self._generate_error(engine, request)))
        worker.start()
        self.assertTrue(runtime.started.wait(1))
        engine.cancel("req")
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertIsInstance(result[0], GenerationCancelled)
        self.assertEqual(engine.snapshot()["state"], "ready")
        engine.close()
        engine.close()
        self.assertEqual(runtime.closed, 1)

    @staticmethod
    def _generate_error(engine, request):
        try:
            engine.generate(request)
        except BaseException as exc:
            return exc
        return None
