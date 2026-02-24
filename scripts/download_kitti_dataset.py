#!/usr/bin/env python3
"""Download KITTI object detection dataset assets with a custom output folder.

By default this script downloads and extracts:
- training left color images (`data_object_image_2.zip`)
- training labels (`data_object_label_2.zip`)

It stores the archives and extracted contents under ``--output-dir``.
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path


IMAGE_ARCHIVE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip"
LABEL_ARCHIVE_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip"


@dataclass(frozen=True)
class DatasetComponent:
    name: str
    url: str
    expected_dir: str


COMPONENTS = {
    "images": DatasetComponent(
        name="images",
        url=IMAGE_ARCHIVE_URL,
        expected_dir="training/image_2",
    ),
    "labels": DatasetComponent(
        name="labels",
        url=LABEL_ARCHIVE_URL,
        expected_dir="training/label_2",
    ),
}


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f}{unit}"
        size /= 1024.0
    return f"{num_bytes}B"


def download_file(url: str, dest: Path, timeout: int = 60, force: bool = False) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"SKIP download (exists): {dest}")
        return

    tmp_dest = dest.with_suffix(dest.suffix + ".part")
    if tmp_dest.exists():
        tmp_dest.unlink()

    print(f"Downloading: {url}")
    print(f"Destination: {dest}")

    start = time.time()
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            total = response.headers.get("Content-Length")
            total_bytes = int(total) if total and total.isdigit() else None

            downloaded = 0
            chunk_size = 1024 * 1024
            last_report = 0.0

            with tmp_dest.open("wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_report >= 0.5:
                        if total_bytes:
                            pct = downloaded / total_bytes * 100
                            msg = f"  {pct:6.2f}% ({human_size(downloaded)} / {human_size(total_bytes)})"
                        else:
                            msg = f"  {human_size(downloaded)} downloaded"
                        print(msg, end="\r", flush=True)
                        last_report = now

            if total_bytes:
                print(f"  100.00% ({human_size(downloaded)} / {human_size(total_bytes)})")
            else:
                print(f"  Completed ({human_size(downloaded)})")

        tmp_dest.replace(dest)
        elapsed = time.time() - start
        speed = downloaded / elapsed if elapsed > 0 else 0
        print(f"Done: {dest} in {elapsed:.1f}s ({human_size(int(speed))}/s)")
    except (urllib.error.URLError, TimeoutError) as exc:
        if tmp_dest.exists():
            tmp_dest.unlink(missing_ok=True)
        raise RuntimeError(f"Download failed for {url}: {exc}") from exc


def extract_zip(archive_path: Path, output_dir: Path, force_extract: bool = False) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"Archive not found: {archive_path}")

    print(f"Extracting: {archive_path.name} -> {output_dir}")
    with zipfile.ZipFile(archive_path, "r") as zf:
        members = zf.namelist()
        if not force_extract:
            # Skip full extraction if all members already exist.
            # Directory entries in zip end with '/' and are ignored.
            files = [m for m in members if not m.endswith("/")]
            if files and all((output_dir / m).exists() for m in files):
                print(f"SKIP extract (all files already exist): {archive_path.name}")
                return
        zf.extractall(output_dir)
    print(f"Extracted: {archive_path.name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download KITTI object detection dataset (images and labels) to a custom folder."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/datasets/kitti").expanduser(),
        help="Target folder for archives and extracted KITTI contents (default: ~/datasets/kitti)",
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=sorted(COMPONENTS.keys()),
        default=["images", "labels"],
        help="Which KITTI assets to download (default: images labels)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download archives only; do not extract zip files.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download archives even if they already exist.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Re-extract archives even if extracted files appear to exist.",
    )
    parser.add_argument(
        "--delete-archives",
        action="store_true",
        help="Delete zip archives after successful extraction.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds for each request (default: 60)",
    )
    return parser.parse_args()


def verify_expected_dirs(output_dir: Path, selected: list[DatasetComponent]) -> None:
    print("\nVerification:")
    for component in selected:
        expected_path = output_dir / component.expected_dir
        status = "OK" if expected_path.exists() else "MISSING"
        print(f"- {component.name:6s}: {status} -> {expected_path}")


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected = [COMPONENTS[name] for name in args.components]

    print("KITTI download script")
    print(f"Output directory: {output_dir}")
    print(f"Components: {', '.join(c.name for c in selected)}")
    print(f"Extract: {'no' if args.no_extract else 'yes'}")
    print()

    archive_paths: list[Path] = []

    try:
        for component in selected:
            archive_name = Path(component.url).name
            archive_path = output_dir / archive_name
            archive_paths.append(archive_path)

            download_file(
                component.url,
                archive_path,
                timeout=args.timeout,
                force=args.force_download,
            )

            if not args.no_extract:
                extract_zip(archive_path, output_dir, force_extract=args.force_extract)

        if not args.no_extract:
            verify_expected_dirs(output_dir, selected)

        if args.delete_archives and not args.no_extract:
            for archive_path in archive_paths:
                if archive_path.exists():
                    archive_path.unlink()
                    print(f"Deleted archive: {archive_path}")

        print("\nCompleted.")
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI entrypoint should report any failure.
        print(f"\nERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
