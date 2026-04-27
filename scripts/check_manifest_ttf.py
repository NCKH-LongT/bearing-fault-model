import os
import csv
import sys
from typing import Dict, Any

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def _parse_cfg(cfg_path: str) -> Dict[str, Any]:
    if yaml is not None:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # Fallback minimal parser for keys we need
    cfg: Dict[str, Any] = {}
    cfg["manifest"] = "data/manifest.csv"
    cfg["exclude_list"] = None
    cfg["temporal_ttf"] = {"train": [0.0, 70.0], "val": [70.0, 80.0], "test": [80.0, 100.1]}
    import re, ast
    with open(cfg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("manifest:"):
            cfg["manifest"] = line.split(":", 1)[1].strip()
        elif line.startswith("exclude_list:"):
            val = line.split(":", 1)[1].strip()
            cfg["exclude_list"] = None if val.lower() in {"null", "none", ""} else val
        elif line.startswith("temporal_ttf:"):
            for key in ("train", "val", "test"):
                i += 1
                if i >= len(lines):
                    break
                sub = lines[i].strip()
                if sub.startswith(f"{key}:"):
                    arr = sub.split(":", 1)[1].strip()
                    try:
                        cfg["temporal_ttf"][key] = ast.literal_eval(arr)
                    except Exception:
                        pass
        i += 1
    return cfg


def main(cfg_path: str, override: list = None):
    cfg = _parse_cfg(cfg_path)

    manifest = cfg["manifest"]
    exclude_list = cfg.get("exclude_list")
    ttf_cfg = cfg.get("temporal_ttf", {"train": [0.0, 70.0], "val": [70.0, 80.0], "test": [80.0, 100.1]})
    if override and len(override) == 6:
        ttf_cfg = {
            "train": [float(override[0]), float(override[1])],
            "val": [float(override[2]), float(override[3])],
            "test": [float(override[4]), float(override[5])],
        }

    if not os.path.exists(manifest):
        print(f"Manifest not found: {manifest}")
        return 1

    exclude = set()
    if exclude_list and os.path.exists(exclude_list):
        with open(exclude_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    exclude.add(line)

    rows = []
    with open(manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize header keys to handle BOM/quotes
        def norm(k: str) -> str:
            return k.replace("\ufeff", "").strip().strip('"')
        normalize = {k: norm(k) for k in reader.fieldnames or []}
        for r in reader:
            rr = {normalize.get(k, k): v for k, v in r.items()}
            fn = rr.get("file")
            if fn in exclude:
                continue
            try:
                p = float((rr.get("ttf_percent") or "").strip() or 0.0)
            except Exception:
                p = 0.0
            cls = (rr.get("fault_type") or "").strip().lower()
            rows.append({"ttf": p, "cls": cls})

    labels = sorted(set(r["cls"] for r in rows))
    print("Labels found:", labels)

    def count_range(lo, hi):
        from collections import Counter
        c = Counter()
        for r in rows:
            if float(lo) <= r["ttf"] < float(hi):
                c[r["cls"]] += 1
        return c

    for name in ["train", "val", "test"]:
        lo, hi = ttf_cfg.get(name, [0, 0])
        c = count_range(lo, hi)
        total = sum(c.values())
        print(f"{name}: [{lo},{hi}) total={total} -> {dict(c)}")

    return 0


if __name__ == "__main__":
    cfg = sys.argv[1] if len(sys.argv) > 1 else "configs/best_temporal.yaml"
    override = sys.argv[2:8] if len(sys.argv) >= 8 else None
    sys.exit(main(cfg, override))
