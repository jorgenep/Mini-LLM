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

    def test_xpu_smoke_report_is_complete_and_controls_training(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "xpu.json"
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/xpu_smoke_test.py"), "--output", str(output)],
                check=False,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text())
            gate_names = {gate["name"] for gate in report["gates"]}
            self.assertIn("torch_xpu_available", gate_names)
            self.assertIn("bf16_gemm", gate_names)
            self.assertIs(report["training_allowed"], report["passed"])
            if report["passed"]:
                self.assertIsNone(report["blocker"])
                self.assertEqual(result.returncode, 0)
            else:
                self.assertIsNotNone(report["blocker"])
                self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
