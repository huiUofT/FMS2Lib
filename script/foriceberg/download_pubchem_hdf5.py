#!/usr/bin/env python3
"""Download the PubChem-by-formula HDF5 used by ICEBERG retrieval examples."""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

from foriceberg_paths import data_dir

URL = "https://zenodo.org/records/15529765/files/pubchem_formulae_inchikey.hdf5"
DEFAULT_OUT = data_dir() / "pubchem_formulae_inchikey.hdf5"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download pubchem_formulae_inchikey.hdf5 from Zenodo."
    )
    parser.add_argument("--url", default=URL, help=f"Default: {URL}")
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Default: {DEFAULT_OUT}",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the output file already exists.",
    )
    return parser.parse_args()


def _format_size(n_bytes: int) -> str:
    value = float(n_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{n_bytes} B"


def download(url: str, out_path: Path, force: bool) -> None:
    out_path = out_path.expanduser().resolve()
    if out_path.exists() and not force:
        print(f"Already exists: {out_path}")
        print("Use --force to re-download.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_suffix(out_path.suffix + ".part")

    print(f"Downloading: {url}")
    print(f"Destination: {out_path}")

    with urllib.request.urlopen(url) as response:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        last_reported = -1
        with part_path.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                if total:
                    percent = int(downloaded * 100 / total)
                    if percent != last_reported and percent % 5 == 0:
                        print(
                            f"Progress: {percent}% "
                            f"({_format_size(downloaded)} / {_format_size(total)})",
                            flush=True,
                        )
                        last_reported = percent

    part_path.replace(out_path)
    print(f"Downloaded: {out_path} ({_format_size(out_path.stat().st_size)})")


def main() -> None:
    args = parse_args()
    try:
        download(args.url, args.out, args.force)
    except KeyboardInterrupt:
        print("\nDownload interrupted.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
