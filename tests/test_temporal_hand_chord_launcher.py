"""Launcher collision protection using a local fake training executable."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / "script/train_temporal_hand_chord.sh"


class TemporalHandChordLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "experiment"
        self.calls = self.root / "calls.jsonl"
        executable = self.root / "fake_python"
        executable.write_text(
            f"#!{sys.executable}\n"
            "import json, os, pathlib, sys, time\n"
            "with open(os.environ['FAKE_TRAIN_CALLS'], 'a') as handle:\n"
            "    handle.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "output = pathlib.Path(next(arg.split('=', 1)[1] for arg in sys.argv if arg.startswith('hydra.run.dir=')))\n"
            "(output / 'chord.pt').write_bytes(b'fake checkpoint')\n"
            "print('fake training output', flush=True)\n"
            "time.sleep(float(os.environ.get('FAKE_TRAIN_DELAY', '0')))\n"
        )
        executable.chmod(0o755)
        self.environment = dict(os.environ, DICE_RL_PYTHON=str(executable), FAKE_TRAIN_CALLS=str(self.calls))

    def launch(self):
        return subprocess.run(
            ["bash", str(LAUNCHER), str(self.output)], env=self.environment,
            capture_output=True, text=True, timeout=10,
        )

    def recorded_calls(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def snapshot(self):
        return {
            str(path.relative_to(self.output)): path.read_bytes() if path.is_file() else None
            for path in self.output.rglob("*")
        }

    def test_fresh_precreated_run_executes_both_tasks_once(self):
        self.output.mkdir()
        manifest = self.output / "run_manifest.json"
        manifest.write_text('{"seed": 42}')
        launcher_log = self.output / "launcher.log"
        launcher_log.write_text("orchestrator log\n")
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = self.recorded_calls()
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            [next(arg for arg in call if arg.startswith("--config-dir=")).rsplit("/", 1)[1] for call in calls],
            ["pick_bucket", "fold_glasses"],
        )
        self.assertEqual(manifest.read_text(), '{"seed": 42}')
        self.assertEqual(launcher_log.read_text(), "orchestrator log\n")
        self.assertTrue((self.output / "status").read_text().startswith("completed exit=0"))
        for task in ("pick_bucket", "fold_glasses"):
            self.assertEqual((self.output / task / "chord.pt").read_bytes(), b"fake checkpoint")
            self.assertIn("fake training output", (self.output / task / "train.log").read_text())

    def test_rerun_refuses_and_preserves_all_outputs(self):
        first = self.launch()
        self.assertEqual(first.returncode, 0, first.stderr)
        snapshot = self.snapshot()
        second = self.launch()
        self.assertNotEqual(second.returncode, 0, "Reusing the experiment directory must fail")
        self.assertIn("refus", second.stderr.lower())
        self.assertEqual(self.snapshot(), snapshot)
        self.assertEqual(len(self.recorded_calls()), 2)

    def test_partial_artifacts_refuse_before_writing_any_files(self):
        for artifact in ("status", "launcher.pid", "pick_bucket", "fold_glasses"):
            for kind in ("file", "directory"):
                with self.subTest(artifact=artifact, kind=kind):
                    self.output = self.root / f"{artifact}-{kind}"
                    self.output.mkdir()
                    self.calls.unlink(missing_ok=True)
                    path = self.output / artifact
                    if kind == "file":
                        path.write_text("existing artifact")
                    else:
                        path.mkdir()
                    snapshot = self.snapshot()
                    result = self.launch()
                    self.assertNotEqual(result.returncode, 0)
                    self.assertEqual(self.snapshot(), snapshot)
                    self.assertEqual(self.recorded_calls(), [])
                    path.unlink() if kind == "file" else path.rmdir()

    def test_reservation_remains_even_if_other_outputs_are_removed(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        # A permanent reservation must also protect a cleaned/archived run.
        for name in ("status", "launcher.pid", "pick_bucket", "fold_glasses"):
            path = self.output / name
            shutil.rmtree(path) if path.is_dir() else path.unlink()
        snapshot = self.snapshot()
        second = self.launch()
        self.assertNotEqual(second.returncode, 0)
        self.assertEqual(self.snapshot(), snapshot)
        self.assertEqual(len(self.recorded_calls()), 2)

    def test_concurrent_launch_allows_only_one_training_sequence(self):
        self.environment["FAKE_TRAIN_DELAY"] = "0.1"
        processes = [
            subprocess.Popen(
                ["bash", str(LAUNCHER), str(self.output)], env=self.environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            ) for _ in range(2)
        ]
        try:
            results = [process.communicate(timeout=10) for process in processes]
            self.assertEqual(sorted(process.returncode for process in processes), [0, 1], results)
            self.assertEqual(len(self.recorded_calls()), 2)
            self.assertTrue((self.output / "status").read_text().startswith("completed exit=0"))
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


if __name__ == "__main__":
    unittest.main()
