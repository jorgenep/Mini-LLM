import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class Phase0PreflightTests(unittest.TestCase):
    def test_environment_diagnostic_writes_valid_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "environment.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/environment_diagnostic.py"), "--output", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(output.read_text())
            self.assertIn("pytorch", report)
            self.assertIn("pci_accelerators", report)

    def test_xpu_smoke_refuses_when_xpu_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "xpu.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/xpu_smoke_test.py"), "--output", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text())
            self.assertFalse(report["passed"])
            self.assertFalse(report["training_allowed"])
            self.assertIsNotNone(report["blocker"])
            self.assertIn("torch_xpu_available", {gate["name"] for gate in report["gates"]})


if __name__ == "__main__":
    unittest.main()
