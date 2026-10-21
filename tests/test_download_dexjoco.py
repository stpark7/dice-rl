"""Archive installation must verify complete task layouts and reject unsafe data."""

import hashlib
import io
import json
import os
import subprocess
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from script.download_dexjoco import MANIFEST_VERSION, install_archive, main


class DownloadDexjocoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "task.tar"
        self.destination = self.root / "data" / "dexjoco"
        self.task = "pick_bucket"

    def make_archive(self, extra=None, omit=None):
        files = {
            "episodes.json": b"{}",
            "ph_pretrain/train.npz": b"training-data",
            "ph_pretrain/normalization.npz": b"normalization",
            "ph_finetune/train.npz": b"finetuning-data",
            "ph_finetune/normalization.npz": b"normalization",
        }
        if omit:
            del files[omit]
        with tarfile.open(self.archive, "w") as output:
            for name, data in files.items():
                info = tarfile.TarInfo(f"{self.task}-img/{name}")
                info.size = len(data)
                output.addfile(info, io.BytesIO(data))
            if extra is not None:
                output.addfile(extra)
        return hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def test_install_preserves_relative_layout_and_is_repeatable(self):
        checksum = self.make_archive()
        result = install_archive(self.archive, self.destination, self.task, checksum)
        self.assertEqual(result, self.destination / "pick_bucket-img")
        self.assertTrue((result / "episodes.json").is_file())
        self.assertEqual((result / "ph_pretrain/train.npz").read_bytes(), b"training-data")
        self.assertEqual((result / "ph_finetune/train.npz").read_bytes(), b"finetuning-data")
        receipt = json.loads((result / ".hf_release.json").read_text())
        self.assertEqual(receipt["sha256"], checksum)
        self.assertEqual(install_archive(self.archive, self.destination, self.task, checksum), result)

    def test_corruption_does_not_create_dataset(self):
        checksum = self.make_archive()
        with self.archive.open("ab") as output:
            output.write(b"corrupted")
        with self.assertRaisesRegex(ValueError, "checksum"):
            install_archive(self.archive, self.destination, self.task, checksum)
        self.assertFalse((self.destination / "pick_bucket-img").exists())

    def test_path_traversal_and_absolute_paths_are_rejected(self):
        for name in ["pick_bucket-img/../../escaped", "/tmp/escaped", "other-img/file"]:
            with self.subTest(name=name):
                checksum = self.make_archive(extra=tarfile.TarInfo(name))
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    install_archive(self.archive, self.destination, self.task, checksum)
                self.assertFalse((self.destination / "pick_bucket-img").exists())

    def test_symbolic_links_are_rejected(self):
        info = tarfile.TarInfo("pick_bucket-img/link")
        info.type = tarfile.SYMTYPE
        info.linkname = "/tmp"
        checksum = self.make_archive(extra=info)
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            install_archive(self.archive, self.destination, self.task, checksum)

    def test_missing_file_never_installs_partial_dataset(self):
        checksum = self.make_archive(omit="ph_finetune/train.npz")
        with self.assertRaisesRegex(ValueError, "Missing"):
            install_archive(self.archive, self.destination, self.task, checksum)
        self.assertFalse((self.destination / "pick_bucket-img").exists())

    def test_existing_local_dataset_is_not_overwritten(self):
        checksum = self.make_archive()
        existing = self.destination / "pick_bucket-img"
        existing.mkdir(parents=True)
        (existing / "local.txt").write_text("keep me")
        with self.assertRaises(FileExistsError):
            install_archive(self.archive, self.destination, self.task, checksum)
        self.assertEqual((existing / "local.txt").read_text(), "keep me")

    def run_cli(self, manifest, arguments=None, select_task=True, archives=None):
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download") as fetch:
            api.return_value.dataset_info.return_value = SimpleNamespace(sha="fixed-commit")
            fetch.side_effect = lambda filename, **kwargs: str(
                manifest_path if filename == "manifest.json" else (archives or {}).get(filename, self.archive)
            )
            main(["--data-dir", str(self.root / "data")]
                 + (["--task", self.task] if select_task else [])
                 + (arguments or []))
            return fetch.call_args_list

    def release_manifest(self):
        checksum = self.make_archive()
        return {"format_version": MANIFEST_VERSION, "tasks": {self.task: {
            "archive": "archives/pick_bucket-img.tar", "sha256": checksum,
            "archive_bytes": self.archive.stat().st_size,
        }}}

    def test_cli_defaults_to_manifest_tasks_and_new_repository(self):
        tasks = ("click_mouse", "fold_glasses", "hammer_nail", "pick_bucket",
                 "pinch_tongs", "water_plant")
        for selected in (tasks, ("click_mouse", "water_plant")):
            with self.subTest(tasks=selected):
                manifest = {"format_version": MANIFEST_VERSION, "tasks": {}}
                archives = {}
                for task in selected:
                    self.task = task
                    entry = self.release_manifest()["tasks"][task]
                    entry["archive"] = f"archives/{task}-img.tar"
                    path = self.root / f"{task}.tar"
                    path.write_bytes(self.archive.read_bytes())
                    archives[entry["archive"]] = path
                    manifest["tasks"][task] = entry
                calls = self.run_cli(manifest, select_task=False, archives=archives)
                self.assertEqual([call.kwargs["filename"] for call in calls],
                                 ["manifest.json"] + list(archives))
                self.assertTrue(all(call.kwargs["repo_id"] ==
                                    "robopark/dexjoco-single-arm-data" for call in calls))

    def test_cli_missing_selected_task_errors_before_archive_download(self):
        manifest = self.release_manifest()
        with patch("huggingface_hub.hf_hub_download") as fetch:
            manifest_path = self.root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            fetch.return_value = str(manifest_path)
            with patch("huggingface_hub.HfApi") as api:
                api.return_value.dataset_info.return_value = SimpleNamespace(sha="fixed")
                with self.assertRaisesRegex(ValueError, "not available.*bimanual_hanoi"):
                    main(["--task", "pick_bucket", "--task", "bimanual_hanoi"])
            self.assertEqual(fetch.call_count, 1)

    def test_cli_selects_multiple_tasks_with_grouped_or_repeated_flags(self):
        manifest = {"format_version": MANIFEST_VERSION, "tasks": {}}
        archives = {}
        for task in ("water_plant", "pick_bucket", "hammer_nail"):
            self.task = task
            entry = self.release_manifest()["tasks"][task]
            entry["archive"] = f"archives/{task}-img.tar"
            path = self.root / f"{task}.tar"
            path.write_bytes(self.archive.read_bytes())
            archives[entry["archive"]] = path
            manifest["tasks"][task] = entry
        for arguments in (
                ["--task", "hammer_nail", "water_plant"],
                ["--task", "hammer_nail", "--task", "water_plant"],
                ["--task", "hammer_nail", "water_plant", "--task", "hammer_nail"]):
            with self.subTest(arguments=arguments):
                calls = self.run_cli(manifest, arguments, select_task=False, archives=archives)
                self.assertEqual([call.kwargs["filename"] for call in calls],
                                 ["manifest.json", "archives/hammer_nail-img.tar",
                                  "archives/water_plant-img.tar"])
                self.assertTrue((self.destination / "hammer_nail-img").is_dir())
                self.assertTrue((self.destination / "water_plant-img").is_dir())
                self.assertFalse((self.destination / "pick_bucket-img").exists())

    def test_shell_routes_default_selector_and_flags_without_network(self):
        stub = self.root / "python"
        stub.write_text('#!/bin/bash\nprintf "%s\\0" "$@" > "$ARGUMENT_CAPTURE"\n')
        stub.chmod(0o755)
        capture = self.root / "arguments"
        env = dict(os.environ, PATH=str(self.root) + os.pathsep + os.environ["PATH"],
                   ARGUMENT_CAPTURE=str(capture))
        env.pop("DICE_RL_DATA_DIR", None)
        env.pop("DICE_RL_LOG_DIR", None)
        script = Path(__file__).resolve().parents[1] / "script/download_hf.sh"
        for arguments, forwarded in (([], []), (["dexjoco", "--task", "click_mouse"],
                                                ["--task", "click_mouse"]),
                                     (["--task", "hammer_nail", "water_plant"],
                                      ["--task", "hammer_nail", "water_plant"])):
            with self.subTest(arguments=arguments):
                result = subprocess.run(["bash", str(script)] + arguments, env=env,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                actual = capture.read_bytes().decode().split("\0")[:-1]
                self.assertEqual(actual, [str(script.with_name("download_dexjoco.py"))]
                                 + forwarded)

    def test_cli_data_directory_defaults_to_environment_or_repository(self):
        manifest = self.release_manifest()
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        repository = Path(__file__).resolve().parents[1]
        for configured in (None, str(self.root / "configured")):
            with self.subTest(configured=configured), patch.dict(os.environ):
                os.environ.pop("DICE_RL_DATA_DIR", None)
                if configured:
                    os.environ["DICE_RL_DATA_DIR"] = configured
                with patch("huggingface_hub.HfApi") as api, patch(
                        "huggingface_hub.hf_hub_download") as fetch, patch(
                        "script.download_dexjoco.install_archive") as install:
                    api.return_value.dataset_info.return_value = SimpleNamespace(sha="fixed")
                    fetch.side_effect = lambda filename, **kwargs: str(
                        manifest_path if filename == "manifest.json" else self.archive)
                    main([])
                    expected = Path(configured) if configured else repository / "data_dir"
                    self.assertEqual(install.call_args.args[1], expected / "dexjoco")

    def test_cli_retains_explicit_legacy_repository_and_task(self):
        self.task = "bimanual_hanoi"
        calls = self.run_cli(self.release_manifest(),
                             ["--repo-id", "robopark/dexjoco-image-data"])
        self.assertEqual(calls[-1].kwargs["repo_id"], "robopark/dexjoco-image-data")
        self.assertTrue((self.destination / "bimanual_hanoi-img").is_dir())

    def test_cli_pins_all_downloads_and_deduplicates_selected_tasks(self):
        calls = self.run_cli(self.release_manifest(), ["--task", self.task])
        self.assertEqual(len(calls), 2)
        self.assertEqual([call.kwargs["filename"] for call in calls],
                         ["manifest.json", "archives/pick_bucket-img.tar"])
        for call in calls:
            self.assertEqual(call.kwargs["revision"], "fixed-commit")
            self.assertEqual(call.kwargs["repo_type"], "dataset")

    def test_cli_rejects_incorrect_archive_size_before_installing(self):
        manifest = self.release_manifest()
        manifest["tasks"][self.task]["archive_bytes"] += 1
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            self.run_cli(manifest)
        self.assertFalse((self.destination / "pick_bucket-img").exists())

    def test_cli_rejects_unknown_manifest_version(self):
        manifest = self.release_manifest()
        manifest["format_version"] = 99
        with self.assertRaisesRegex(ValueError, "manifest version"):
            self.run_cli(manifest)


if __name__ == "__main__":
    unittest.main()
