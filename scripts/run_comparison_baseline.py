from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent
DEFS_DIR = ROOT / "comparison_baselines" / "definitions"


def load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_definitions() -> List[Dict]:
    items: List[Dict] = []
    for path in sorted(DEFS_DIR.glob("*.json")):
        data = load_json(path)
        data["_path"] = str(path.relative_to(ROOT))
        items.append(data)
    return items


def get_definition(baseline_id: str) -> Dict:
    path = DEFS_DIR / f"{baseline_id}.json"
    if not path.exists():
        known = ", ".join(d["id"] for d in list_definitions())
        raise SystemExit(f"Unknown baseline '{baseline_id}'. Known: {known}")
    data = load_json(path)
    data["_path"] = str(path.relative_to(ROOT))
    return data


def run_cmd(cmd: List[str], dry_run: bool) -> int:
    print("$", " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=str(ROOT))


def deep_ckpt_for_protocol(proto_cfg: Dict) -> str:
    ckpt = proto_cfg.get("ckpt")
    if ckpt:
        return ckpt
    raise SystemExit(f"Missing ckpt in baseline definition for config: {proto_cfg['config']}")


def show_definition(defn: Dict) -> None:
    print(f"id: {defn['id']}")
    print(f"display_name: {defn.get('display_name', '')}")
    print(f"family: {defn.get('family', '')}")
    print(f"definition_file: {defn.get('_path', '')}")
    print(f"summary: {defn.get('summary', '')}")
    print("protocols:")
    for name, cfg in (defn.get("protocols", {}) or {}).items():
        print(f"  - {name}:")
        print(f"      config: {cfg.get('config', '')}")
        if cfg.get("ckpt"):
            print(f"      ckpt: {cfg['ckpt']}")
        if cfg.get("primary_report"):
            print(f"      report: {cfg['primary_report']}")
        if cfg.get("notes"):
            print(f"      notes: {cfg['notes']}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="List registered comparison baselines")
    ap.add_argument("--show", action="store_true", help="Show the selected baseline definition")
    ap.add_argument("--baseline", help="Baseline id, e.g. svm_vib8")
    ap.add_argument("--protocol", help="Protocol key inside the selected baseline, e.g. stratified/temporal/fullrange")
    ap.add_argument("--action", default="train_eval", choices=["train", "eval", "train_eval"], help="Action to run")
    ap.add_argument("--agg", default="mean", choices=["mean", "vote"], help="Aggregation used for deep eval")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without executing")
    args = ap.parse_args()

    if args.list:
        for d in list_definitions():
            print(f"{d['id']}: {d.get('display_name', '')}")
        return 0

    if not args.baseline:
        raise SystemExit("--baseline is required unless --list is used.")

    defn = get_definition(args.baseline)

    if args.show:
        show_definition(defn)
        return 0

    if not args.protocol:
        raise SystemExit("--protocol is required when running a baseline.")

    protocols = defn.get("protocols", {}) or {}
    if args.protocol not in protocols:
        known = ", ".join(protocols.keys())
        raise SystemExit(f"Unknown protocol '{args.protocol}' for baseline '{args.baseline}'. Known: {known}")

    proto_cfg = protocols[args.protocol]
    runner = defn.get("runner", {}) or {}
    runner_type = (runner.get("type") or "").strip().lower()

    if runner_type == "classical":
        entry = runner["train_eval_entry"]
        return run_cmd([args.python, entry, "--config", proto_cfg["config"]], args.dry_run)

    if runner_type == "deep":
        train_entry = runner["train_entry"]
        eval_entry = runner["eval_entry"]
        rc = 0
        if args.action in {"train", "train_eval"}:
            rc = run_cmd([args.python, train_entry, "--config", proto_cfg["config"]], args.dry_run)
            if rc != 0:
                return rc
        if args.action in {"eval", "train_eval"}:
            ckpt = deep_ckpt_for_protocol(proto_cfg)
            cmd = [args.python, eval_entry, "--config", proto_cfg["config"], "--ckpt", ckpt]
            if args.agg != "mean":
                cmd += ["--agg", args.agg]
            rc = run_cmd(cmd, args.dry_run)
        return rc

    raise SystemExit(f"Unsupported runner type: {runner_type}")


if __name__ == "__main__":
    raise SystemExit(main())
