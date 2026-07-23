import os
from pathlib import Path
import subprocess
import sys
import unittest


class ServerImportBoundaryTest(unittest.TestCase):
    def test_server_imports_without_runtime_extras(self):
        root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root / "src")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        result = subprocess.run(
            [sys.executable, "-S", "-c", "import sparseflow.server, sparseflow.serving_types; print('ok')"],
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "ok")
