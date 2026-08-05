#!/usr/bin/env python3
"""Convert manifest CSV signals to resumable float32 NPY files for mmap loading."""

from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def convert_one(src_text: str, dst_text: str, force: bool) -> tuple[str, str, int, int]:
    src = Path(src_text)
    dst = Path(dst_text)
    if dst.is_file() and not force:
        arr = np.load(dst, mmap_mode="r")
        return src.name, "cached", int(arr.shape[0]), int(arr.shape[1])

    arr = np.loadtxt(src, delimiter=",", dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 4:
        raise ValueError(f"Expected at least four columns in {src}, got {arr.shape}")
    arr = np.ascontiguousarray(arr[:, :4], dtype=np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + f".{os.getpid()}.tmp")
    with tmp.open("wb") as f:
        np.save(f, arr, allow_pickle=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, dst)
    return src.name, "written", int(arr.shape[0]), int(arr.shape[1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/manifest.csv")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data_cache/npy")
    parser.add_argument("--workers", type=int, default=2, help="Parallel CSV parsers; 2 is safe for ~32 GB RAM.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest = ROOT / args.manifest
    data_dir = ROOT / args.data_dir
    output_dir = ROOT / args.output_dir
    with manifest.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    jobs: list[tuple[str, str, bool]] = []
    for row in rows:
        filename = (row.get("file") or "").strip()
        if not filename:
            continue
        src = data_dir / filename
        dst = output_dir / f"{src.stem}.npy"
        if not src.is_file():
            raise SystemExit(f"Missing source: {src.relative_to(ROOT)}")
        jobs.append((str(src), str(dst), bool(args.force)))

    output_dir.mkdir(parents=True, exist_ok=True)
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [pool.submit(convert_one, *job) for job in jobs]
        for future in as_completed(futures):
            name, status, rows_count, cols_count = future.result()
            done += 1
            print(f"[{done}/{len(jobs)}] {status:7s} {name}: {rows_count}x{cols_count}", flush=True)

    print(f"Cache ready: {output_dir.relative_to(ROOT)} ({len(jobs)} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
