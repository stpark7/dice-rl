"""Download versioned Dexjoco image datasets and verify them before installation."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile


DEFAULT_REPO = "robopark/dexjoco-single-arm-data"
TASKS = (
    "click_mouse", "fold_glasses", "hammer_nail", "pick_bucket",
    "pinch_tongs", "water_plant",
)
LEGACY_TASKS = (
    "bimanual_assembly",
    "bimanual_hanoi", "bimanual_microwave_cook",
)
# Version 1 kept images in a sibling images.zarr; version 2 stores them inside
# the NPZ files as images_<camera> arrays, which the loader now requires.
MANIFEST_VERSION = 2
REQUIRED_FILES = (
    "episodes.json",
    "ph_pretrain/train.npz", "ph_pretrain/normalization.npz",
    "ph_finetune/train.npz", "ph_finetune/normalization.npz",
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_layout(path):
    missing = [name for name in REQUIRED_FILES if not (Path(path) / name).is_file()]
    if missing:
        raise ValueError(f"Missing dataset files in {path}: {missing}")


def install_archive(archive, dataset_root, task, expected_sha256):
    """Verify, safely extract, and atomically install one task without overwriting.

    dataset_root is the dexjoco directory, not DICE_RL_DATA_DIR itself. A receipt
    makes an unchanged completed installation repeatable; it is not a substitute
    for an audit if the extracted files have subsequently been edited.
    """
    if task not in TASKS + LEGACY_TASKS:
        raise ValueError(f"Unknown Dexjoco task: {task}")
    if sha256_file(archive) != expected_sha256:
        raise ValueError(f"SHA-256 checksum mismatch: {archive}")
    dataset_root = Path(dataset_root)
    directory_name = f"{task}-img"
    target = dataset_root / directory_name
    receipt_path = target / ".hf_release.json"
    if target.exists():
        if receipt_path.is_file():
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("sha256") == expected_sha256:
                check_layout(target)
                return target
        raise FileExistsError(
            f"Refusing to overwrite {target}. Choose a different --data-dir "
            "or move the existing dataset before downloading."
        )
    dataset_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{task}-", dir=dataset_root) as temp:
        staging = Path(temp)
        with tarfile.open(archive, "r:*") as source:
            for member in source:
                name = PurePosixPath(member.name)
                if (name.is_absolute() or ".." in name.parts or not name.parts
                        or name.parts[0] != directory_name
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f"Unsafe archive member: {member.name}")
                destination = staging.joinpath(*name.parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as data, destination.open("xb") as output:
                        shutil.copyfileobj(data, output)
        extracted = staging / directory_name
        check_layout(extracted)
        (extracted / ".hf_release.json").write_text(
            json.dumps({"task": task, "sha256": expected_sha256}, indent=2) + "\n"
        )
        extracted.rename(target)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main", help="HF commit, tag, or branch")
    parser.add_argument("--task", action="extend", nargs="+", choices=TASKS + LEGACY_TASKS,
                        help="One or more tasks; may be repeated; defaults to all tasks in the manifest")
    parser.add_argument("--data-dir", default=os.environ.get("DICE_RL_DATA_DIR") or
                        str(Path(__file__).resolve().parents[1] / "data_dir"))
    parser.add_argument("--cache-dir", default=None, help="Optional HF archive cache")
    args = parser.parse_args(argv)
    if not args.data_dir:
        parser.error("Set DICE_RL_DATA_DIR or pass --data-dir")

    # Public releases need no token; private legacy releases use standard HF login.
    from huggingface_hub import HfApi, hf_hub_download

    revision = HfApi().dataset_info(args.repo_id, revision=args.revision).sha
    common = dict(repo_id=args.repo_id, repo_type="dataset", revision=revision,
                  cache_dir=args.cache_dir)
    manifest = json.loads(Path(hf_hub_download(filename="manifest.json", **common)).read_text())
    if manifest.get("format_version") != MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported Dexjoco release manifest version {manifest.get('format_version')}; "
            f"this code installs version {MANIFEST_VERSION} (images inside the NPZ files)"
        )
    print(f"Dataset: {args.repo_id}@{revision}", flush=True)
    selected_tasks = list(dict.fromkeys(args.task or manifest["tasks"]))
    for task in selected_tasks:
        if task not in TASKS + LEGACY_TASKS:
            raise ValueError(f"Unsupported Dexjoco task in manifest: {task}")
        if task not in manifest["tasks"]:
            raise ValueError(
                f"Task not available in {args.repo_id}: {task}. "
                f"Available tasks: {', '.join(manifest['tasks'])}"
            )
    for task in dict.fromkeys(args.task or manifest["tasks"]):
        entry = manifest["tasks"][task]
        print(f"Downloading and checking {task} ...", flush=True)
        archive = hf_hub_download(filename=entry["archive"], **common)
        if Path(archive).stat().st_size != entry["archive_bytes"]:
            raise ValueError(f"Archive size mismatch: {task}")
        target = install_archive(archive, Path(args.data_dir) / "dexjoco",
                                 task, entry["sha256"])
        print(f"Ready: {target}", flush=True)
    print(f"Complete. Record HF revision {revision} with the training run.")


if __name__ == "__main__":
    main()
