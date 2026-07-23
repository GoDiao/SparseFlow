import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from benchmarks.run_stage7_11_server_acceptance import ServerSession


class FakeHTTPResponse:
    def __init__(self, lines=(), payload=b"{}"):
        self.headers = {"x-request-id": "req_fake"}
        self.status = 200
        self._lines = iter(line if isinstance(line, bytes) else line.encode("utf-8") for line in lines)
        self._payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def readline(self):
        return next(self._lines, b"")

    def read(self):
        return self._payload


class Stage711ServerAcceptanceTest(unittest.TestCase):
    def make_session(self, root: Path, preset: str = "low-memory") -> ServerSession:
        return ServerSession(
            model=root / "model",
            container=root / "container",
            host="127.0.0.1",
            port=8768,
            preset=preset,
            cache_bytes="256MiB",
            context_tokens=2048,
            max_completion_tokens=8,
            log_dir=root / "logs",
        )

    def test_command_is_single_server_process_with_explicit_port(self):
        with tempfile.TemporaryDirectory() as temp:
            session = self.make_session(Path(temp))
            self.assertIn("serve", session.command)
            self.assertIn("--port", session.command)
            self.assertIn("8768", session.command)

    def test_laptop_preset_is_forwarded_to_server_command(self):
        with tempfile.TemporaryDirectory() as temp:
            session = self.make_session(Path(temp), preset="laptop-16gb")
        self.assertIn("laptop-16gb", session.command)

    def test_sse_parser_collects_deltas_and_done(self):
        lines = [
            b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n",
            b"\n",
            b"data: {\"choices\":[{\"delta\":{\"content\":\"Hi\"}}]}\n",
            b"\n",
            b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"length\"}]}\n",
            b"\n",
            b"data: [DONE]\n",
            b"\n",
        ]
        with tempfile.TemporaryDirectory() as temp:
            session = self.make_session(Path(temp))
            with patch(
                "benchmarks.run_stage7_11_server_acceptance.urlrequest.urlopen",
                return_value=FakeHTTPResponse(lines=lines),
            ):
                result = session.stream_request(
                    [{"role": "user", "content": "hello"}],
                    tokens=2,
                )
        self.assertEqual(result["status"], 200)
        self.assertEqual(result["request_id"], "req_fake")
        self.assertEqual(result["text"], "Hi")
        self.assertTrue(result["done"])
        self.assertEqual(len(result["events"]), 4)

    def test_windows_close_uses_process_tree_termination(self):
        with tempfile.TemporaryDirectory() as temp:
            session = self.make_session(Path(temp))
            process = MagicMock()
            process.poll.return_value = None
            process.pid = 4321
            session.process = process
            session._stdout = io.StringIO()
            session._stderr = io.StringIO()
            with patch(
                "benchmarks.run_stage7_11_server_acceptance.os.name",
                "nt",
            ), patch(
                "benchmarks.run_stage7_11_server_acceptance.subprocess.run"
            ) as run:
                session.close()
            command = run.call_args.args[0]
            self.assertEqual(command[:2], ["taskkill", "/PID"])
            self.assertIn("/T", command)
            self.assertIn("/F", command)
            process.wait.assert_called()


if __name__ == "__main__":
    unittest.main()
