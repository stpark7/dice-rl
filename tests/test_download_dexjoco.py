"""Archive installation must preserve image references and reject unsafe data."""

import hashlib
import io
import json
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from script.download_dexjoco import install_archive, main


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
            "images.zarr/.zgroup": b'{"zarr_format": 2}',
            "images.zarr/.zattrs": b'{"complete": true}',
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
        self.assertTrue((result / "ph_pretrain/../images.zarr/.zgroup").is_file())
        self.assertEqual((result / "ph_pretrain/train.npz").read_bytes(), b"training-data")
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

    def test_missing_image_store_never_installs_partial_dataset(self):
        checksum = self.make_archive(omit="images.zarr/.zgroup")
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

    def run_cli(self, manifest, arguments=None):
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))
        with patch("huggingface_hub.HfApi") as api, patch("huggingface_hub.hf_hub_download") as fetch:
            api.return_value.dataset_info.return_value = SimpleNamespace(sha="fixed-commit")
            fetch.side_effect = lambda filename, **kwargs: str(
                manifest_path if filename == "manifest.json" else self.archive
            )
            main(["--data-dir", str(self.root / "data"), "--task", self.task]
                 + (arguments or []))
            return fetch.call_args_list

    def release_manifest(self):
        checksum = self.make_archive()
        return {"format_version": 1, "tasks": {self.task: {
            "archive": "archives/pick_bucket-img.tar", "sha256": checksum,
            "archive_bytes": self.archive.stat().st_size,
        }}}

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
