"""Portable sequential RL launcher checks with no simulator or model loading."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / 'script/train_temporal_hand_chord_rl.sh'


class TemporalHandChordRLLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.codec_root = self.root / 'codec'
        self.output = self.root / 'rl'
        self.calls = self.root / 'calls.jsonl'
        for task in ('pick_bucket', 'fold_glasses'):
            (self.codec_root / task).mkdir(parents=True)
            (self.codec_root / task / 'chord.pt').write_bytes(b'codec')
        executable = self.root / 'fake_python'
        executable.write_text(
            f'#!{sys.executable}\n'
            'import json, os, sys, time\n'
            "record = {'args': sys.argv[1:], 'cwd': os.getcwd(), 'env': {key: os.environ[key] for key in "
            "['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MUJOCO_GL', "
            "'PYOPENGL_PLATFORM', 'DICE_RL_DATA_DIR', 'DICE_RL_LOG_DIR']}}\n"
            "with open(os.environ['FAKE_CALLS'], 'a') as stream: stream.write(json.dumps(record) + '\\n')\n"
            "task = next(arg.rsplit('/', 1)[1] for arg in sys.argv if arg.startswith('--config-dir='))\n"
            "print('fake RL ' + task, flush=True)\n"
            "time.sleep(float(os.environ.get('FAKE_DELAY', '0')))\n"
            "sys.exit(17 if task == os.environ.get('FAKE_FAIL_TASK') else 0)\n"
        )
        executable.chmod(0o755)
        self.environment = dict(os.environ, DICE_RL_PYTHON=str(executable), FAKE_CALLS=str(self.calls))
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'MUJOCO_GL', 'PYOPENGL_PLATFORM', 'DICE_RL_DATA_DIR', 'DICE_RL_LOG_DIR'):
            self.environment.pop(key, None)

    def launch(self):
        return subprocess.run(['bash', str(LAUNCHER), str(self.codec_root), str(self.output)],
                              cwd=self.root, env=self.environment, capture_output=True, text=True, timeout=10)

    def recorded(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def snapshot(self):
        return {str(path.relative_to(self.output)): path.read_bytes() if path.is_file() else None
                for path in self.output.rglob('*')}

    def test_sequential_commands_use_full_reference_configs_without_parameter_reductions(self):
        result = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        records = self.recorded()
        self.assertEqual(len(records), 2)
        for task, record in zip(('pick_bucket', 'fold_glasses'), records):
            self.assertEqual(record['cwd'], str(REPO_ROOT))
            self.assertEqual(record['args'][:3], [
                'script/run.py', '--config-name=ft_temporal_hand_chord_residual_flow_unet_img',
                '--config-dir=' + str(REPO_ROOT / 'cfg/dexjoco/finetune' / task)])
            self.assertEqual(set(record['args'][3:]), {
                'chord_run_dir=' + str(self.codec_root), 'logdir=' + str(self.output / task),
                'hydra.run.dir=' + str(self.output / task), 'device=cuda:0', 'seed=42'})
            self.assertEqual(len(record['args']), 8)
            self.assertEqual(record['env'], {
                'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                'MUJOCO_GL': 'egl', 'PYOPENGL_PLATFORM': 'egl',
                'DICE_RL_DATA_DIR': str(REPO_ROOT / 'data_dir'),
                'DICE_RL_LOG_DIR': str(REPO_ROOT / 'log_dir')})
            self.assertTrue((self.output / task / 'status').read_text().startswith('completed'))
            self.assertIn('fake RL ' + task, (self.output / task / 'train.log').read_text())
            self.assertIn('chord_run_dir=', (self.output / task / 'command.sh').read_text())
        self.assertTrue((self.output / 'status').read_text().startswith('completed exit=0'))

    def test_missing_checkpoint_rejected_before_any_output_artifacts(self):
        (self.codec_root / 'fold_glasses/chord.pt').unlink()
        result = self.launch()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('fold_glasses', result.stderr)
        self.assertFalse(self.output.exists())
        self.assertFalse(self.calls.exists())

    def test_rerun_preserves_outputs_and_partial_artifacts_are_rejected(self):
        self.assertEqual(self.launch().returncode, 0)
        before = self.snapshot()
        self.assertNotEqual(self.launch().returncode, 0)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(self.recorded()), 2)
        for artifact in ('status', 'launcher.pid', 'pick_bucket', 'fold_glasses'):
            self.output = self.root / ('partial-' + artifact)
            self.output.mkdir()
            (self.output / artifact).write_text('preserve')
            before = self.snapshot()
            self.assertNotEqual(self.launch().returncode, 0)
            self.assertEqual(self.snapshot(), before)
            self.assertEqual(len(self.recorded()), 2)

    def test_first_failure_stops_queue_and_propagates_exit_status(self):
        self.environment['FAKE_FAIL_TASK'] = 'pick_bucket'
        result = self.launch()
        self.assertEqual(result.returncode, 17, result.stderr)
        self.assertEqual(len(self.recorded()), 1)
        self.assertTrue((self.output / 'status').read_text().startswith('failed exit=17'))
        self.assertTrue((self.output / 'pick_bucket/status').read_text().startswith('failed exit=17'))
        self.assertFalse((self.output / 'fold_glasses').exists())

    def test_atomic_reservation_allows_only_one_concurrent_queue(self):
        self.environment['FAKE_DELAY'] = '.1'
        processes = [subprocess.Popen(
            ['bash', str(LAUNCHER), str(self.codec_root), str(self.output)], cwd=self.root,
            env=self.environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            for _ in range(2)]
        try:
            outputs = [process.communicate(timeout=10) for process in processes]
            self.assertEqual(sorted(process.returncode for process in processes), [0, 1], outputs)
            self.assertEqual(len(self.recorded()), 2)
        finally:
            for process in processes:
                if process.poll() is None:
                    process.kill()
                    process.communicate()


if __name__ == '__main__':
    unittest.main()
