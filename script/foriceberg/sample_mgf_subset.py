#!/usr/bin/env python3
"""Deterministically sample N MGF spectra for training smoke examples."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from foriceberg_paths import data_dir, repo_root


REQUIRED_KEYS = ("IDENTIFIER", "SMILES", "PRECURSOR_MZ", "ADDUCT", "INSTRUMENT_TYPE")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample a fixed-seed subset of MGF spectra.")
    p.add_argument(
        "--mgf",
        type=Path,
        default=repo_root() / "data" / "FMS2lib" / "F_ICEBERG.mgf",
        help="Source MGF (default: data/FMS2lib/F_ICEBERG.mgf)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=data_dir() / "examples" / "F_ICEBERG_train100.mgf",
        help="Output MGF path",
    )
    p.add_argument("--n", type=int, default=100, help="Number of spectra to sample")
    p.add_argument("--seed", type=int, default=1, help="RNG seed for reproducibility")
    return p.parse_args()


def iter_blocks(path: Path) -> List[Tuple[str, Dict[str, str], int]]:
    blocks: List[Tuple[str, Dict[str, str], int]] = []
    cur_lines: List[str] = []
    meta: Dict[str, str] = {}
    n_peaks = 0
    in_block = False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            stripped = line.strip()
            if stripped == "BEGIN IONS":
                in_block = True
                cur_lines = [line]
                meta = {}
                n_peaks = 0
                continue
            if not in_block:
                continue
            cur_lines.append(line)
            if stripped == "END IONS":
                blocks.append(("\n".join(cur_lines) + "\n", dict(meta), n_peaks))
                in_block = False
                continue
            if "=" in stripped and not stripped[0].isdigit():
                key, value = stripped.split("=", 1)
                meta[key] = value
            else:
                parts = stripped.split()
                if len(parts) >= 2:
                    try:
                        float(parts[0])
                        float(parts[1])
                        n_peaks += 1
                    except ValueError:
                        pass
    return blocks


def is_valid(meta: Dict[str, str], n_peaks: int) -> bool:
    if n_peaks <= 0:
        return False
    return all(meta.get(k) for k in REQUIRED_KEYS)


def main() -> None:
    args = parse_args()
    if not args.mgf.is_file():
        sys.exit(f"Missing source MGF: {args.mgf}")
    if args.n <= 0:
        sys.exit("--n must be positive")

    blocks = iter_blocks(args.mgf)
    valid_idx = [i for i, (_text, meta, n_peaks) in enumerate(blocks) if is_valid(meta, n_peaks)]
    if len(valid_idx) < args.n:
        sys.exit(f"Only {len(valid_idx)} valid spectra available; need {args.n}")

    rng = random.Random(args.seed)
    chosen = sorted(rng.sample(valid_idx, args.n))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for i in chosen:
            handle.write(blocks[i][0])
            if not blocks[i][0].endswith("\n"):
                handle.write("\n")

    ids = [blocks[i][1]["IDENTIFIER"] for i in chosen]
    print(f"Wrote {args.n} spectra -> {args.out}")
    print(f"seed={args.seed}; first_id={ids[0]}; last_id={ids[-1]}")


if __name__ == "__main__":
    main()
