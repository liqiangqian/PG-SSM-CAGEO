import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.run_demo import run_demo


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "manuscript_demo.json"


class DemoSmokeTests(unittest.TestCase):
    def test_repository_root_cli_entrypoint_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [
                    sys.executable,
                    "scripts/run_demo.py",
                    "--config",
                    "configs/manuscript_demo.json",
                    "--epochs",
                    "1",
                    "--output",
                    str(Path(directory) / "cli.json"),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_cpu_demo_runs_and_marks_output_synthetic(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            result = run_demo(CONFIG, epochs=1, output_path=output)
            self.assertTrue(result["synthetic_only"])
            self.assertTrue(result["workflow_verification_only"])
            self.assertFalse(result["recomputes_field_metrics"])
            self.assertEqual(result["device"], "CPU")
            self.assertEqual((result["history_days"], result["horizon_days"]), (28, 7))
            self.assertEqual(result["seed"], 43)
            self.assertGreater(result["scored_test_targets"], 0)
            self.assertEqual(result, json.loads(output.read_text(encoding="utf-8")))

    def test_demo_is_deterministic_for_the_frozen_seed(self):
        with tempfile.TemporaryDirectory() as directory:
            first = run_demo(CONFIG, epochs=1, output_path=Path(directory) / "a.json")
            second = run_demo(CONFIG, epochs=1, output_path=Path(directory) / "b.json")
            self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
