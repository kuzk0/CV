"""Download the Kaggle Russian license-plate characters dataset.

Downloads the published dataset into Datasets/Plates/raw without modifying
classes.names or the detector's separate Cars detection dataset.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import uuid
from pathlib import Path

DATASET_ID = "dimahkiin/russian-license-plate-characters-23-classes"
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "Datasets" / "Plates" / "raw"


def find_kaggle_executable() -> str | None:
    """Find the Kaggle CLI installed in the active Python environment."""
    executable_name = "kaggle.exe" if os.name == "nt" else "kaggle"
    scripts_dir = Path(sysconfig.get_path("scripts"))
    candidate = scripts_dir / executable_name
    if candidate.is_file():
        return str(candidate)
    return shutil.which("kaggle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download the Kaggle dataset of Russian license-plate characters "
            "into Datasets/Plates/raw."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory (default: Datasets/Plates/raw).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the destination directory if it already contains files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    output_dir = output_dir.resolve()

    kaggle_executable = find_kaggle_executable()
    if kaggle_executable is None:
        print(
            "ERROR: Kaggle CLI not found. Activate .venv and run:\n"
            "  python -m pip install -r requirements.txt",
            file=sys.stderr,
        )
        return 2

    if output_dir.exists() and not output_dir.is_dir():
        print(f"ERROR: Destination exists and is not a directory: {output_dir}", file=sys.stderr)
        return 2

    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        print(
            f"ERROR: Destination is not empty: {output_dir}\n"
            "Nothing was overwritten. To replace it intentionally, run with --force.",
            file=sys.stderr,
        )
        return 2

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix=".plates-download-", dir=str(output_dir.parent))
    )
    backup_dir: Path | None = None

    command = [
        kaggle_executable,
        "datasets",
        "download",
        "-d",
        DATASET_ID,
        "--path",
        str(staging_dir),
        "--unzip",
    ]
    print(f"Dataset: https://www.kaggle.com/datasets/{DATASET_ID}")
    print(f"Downloading into temporary directory: {staging_dir}")

    try:
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            print(
                "\nKaggle download failed. Check internet access and Kaggle API credentials.\n"
                "The token should be configured outside this repository; do not commit kaggle.json.",
                file=sys.stderr,
            )
            return result.returncode or 1

        downloaded_files = [p for p in staging_dir.rglob("*") if p.is_file()]
        if not downloaded_files:
            print(
                "ERROR: Kaggle command completed but no files were extracted. "
                "Inspect the Kaggle CLI output and dataset archive.",
                file=sys.stderr,
            )
            return 1

        if output_dir.exists():
            if args.force:
                backup_dir = output_dir.with_name(f"{output_dir.name}.backup")
                if backup_dir.exists():
                    backup_dir = output_dir.with_name(
                        f"{output_dir.name}.backup-{uuid.uuid4().hex[:8]}"
                    )
                output_dir.rename(backup_dir)
            else:
                # Empty destination: remove it before promoting the staged folder.
                output_dir.rmdir()

        staging_dir.rename(output_dir)
        print(f"\nDownload complete: {output_dir}")
        print(f"Extracted files: {len(downloaded_files)}")
        if backup_dir is not None:
            shutil.rmtree(backup_dir, ignore_errors=True)
        print(
            "\nNote: this script preserves the dataset's original layout and does not "
            "generate YOLO bounding-box annotations or overwrite classes.names."
        )
        return 0
    except OSError as exc:
        print(f"ERROR while placing downloaded data: {exc}", file=sys.stderr)
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.rename(output_dir)
        return 1
    finally:
        if staging_dir.exists():
            shutil.rmtree(staging_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
