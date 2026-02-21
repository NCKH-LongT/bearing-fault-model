import os
import re
import csv
import argparse
from fnmatch import fnmatch
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


MACRO_RE = re.compile(r"^\s*macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)")
CLS_RE = re.compile(r"^\s*(healthy|degrading|fault)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", re.IGNORECASE)


def parse_report(report_path: str) -> Tuple[Optional[float], Dict[str, float]]:
    """Parse sklearn.classification_report text to extract macro F1 and per-class F1.
    Returns (macro_f1, {class_name_lower: f1}).
    """
    if not os.path.exists(report_path):
        return None, {}
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            lines = [ln.rstrip("\n") for ln in f]
    except Exception:
        return None, {}

    macro_f1: Optional[float] = None
    per_cls: Dict[str, float] = {}
    for ln in lines:
        m = MACRO_RE.match(ln)
        if m:
            try:
                # precision, recall, f1
                macro_f1 = float(m.group(3))
            except Exception:
                pass
        c = CLS_RE.match(ln)
        if c:
            name = c.group(1).lower()
            try:
                f1 = float(c.group(4))
                per_cls[name] = f1
            except Exception:
                pass
    return macro_f1, per_cls


SEED_SUFFIX_RE = re.compile(r"^(?P<base>.*)_s(?P<seed>\d+)$", re.IGNORECASE)


def split_label_seed(run_name: str) -> Tuple[str, Optional[int]]:
    m = SEED_SUFFIX_RE.match(run_name)
    if m:
        base = m.group("base")
        try:
            seed = int(m.group("seed"))
        except Exception:
            seed = None
        return base, seed
    return run_name, None


def safe_mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (mean(vals) if vals else None)


def safe_std(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return (pstdev(vals) if len(vals) >= 2 else 0.0 if len(vals) == 1 else None)


def scan_runs(base_dir: str, pattern: str, eval_rel: str) -> List[Dict[str, Optional[float]]]:
    rows: List[Dict[str, Optional[float]]] = []
    if not os.path.isdir(base_dir):
        return rows
    for name in sorted(os.listdir(base_dir)):
        full = os.path.join(base_dir, name)
        if not os.path.isdir(full):
            continue
        if not fnmatch(name, pattern):
            continue
        report_path = os.path.join(full, eval_rel)
        macro_f1, per_cls = parse_report(report_path)
        base, seed = split_label_seed(name)
        rows.append({
            "run_dir": name,
            "label_base": base,
            "seed": seed,
            "macro_f1": macro_f1,
            "f1_healthy": per_cls.get("healthy"),
            "f1_degrading": per_cls.get("degrading"),
            "f1_fault": per_cls.get("fault"),
        })
    return rows


def write_csv(path: str, rows: List[Dict[str, Optional[float]]], header: List[str]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in header})


def aggregate_groups(rows: List[Dict[str, Optional[float]]]) -> List[Dict[str, Optional[float]]]:
    by: Dict[str, List[Dict[str, Optional[float]]]] = {}
    for r in rows:
        key = str(r.get("label_base"))
        by.setdefault(key, []).append(r)
    out: List[Dict[str, Optional[float]]] = []
    for key, items in sorted(by.items()):
        mvals = [it.get("macro_f1") for it in items]
        hvals = [it.get("f1_healthy") for it in items]
        dvals = [it.get("f1_degrading") for it in items]
        fvals = [it.get("f1_fault") for it in items]
        out.append({
            "label_base": key,
            "n": len(items),
            "macro_f1_mean": safe_mean(mvals),
            "macro_f1_std": safe_std(mvals),
            "f1_healthy_mean": safe_mean(hvals),
            "f1_healthy_std": safe_std(hvals),
            "f1_degrading_mean": safe_mean(dvals),
            "f1_degrading_std": safe_std(dvals),
            "f1_fault_mean": safe_mean(fvals),
            "f1_fault_std": safe_std(fvals),
        })
    return out


def write_markdown(path: str, groups: List[Dict[str, Optional[float]]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Tổng hợp Ablation\n\n")
        f.write("Các số liệu là mean±std theo seed cho từng nhóm.\n\n")
        for g in groups:
            f.write(f"## {g['label_base']} (n={g['n']})\n")
            def fmt(mu, sd):
                return "NA" if mu is None else (f"{mu:.4f}±{sd:.4f}" if sd is not None else f"{mu:.4f}")
            f.write(f"- Macro‑F1: {fmt(g['macro_f1_mean'], g['macro_f1_std'])}\n")
            f.write(f"- F1 healthy: {fmt(g['f1_healthy_mean'], g['f1_healthy_std'])}\n")
            f.write(f"- F1 degrading: {fmt(g['f1_degrading_mean'], g['f1_degrading_std'])}\n")
            f.write(f"- F1 fault: {fmt(g['f1_fault_mean'], g['f1_fault_std'])}\n\n")


def main():
    ap = argparse.ArgumentParser(description="Aggregate ablation reports into CSV/Markdown")
    ap.add_argument("--base_dir", default="runs/ablations", help="Thư mục gốc chứa các run ablation")
    ap.add_argument("--pattern", default="*", help="Mẫu tên thư mục cần lấy (fnmatch)")
    ap.add_argument("--eval_rel", default=os.path.join("eval", "report.txt"), help="Đường dẫn con tới report.txt")
    ap.add_argument("--out_csv", default=os.path.join("figures", "ablation_per_run.csv"))
    ap.add_argument("--out_group_csv", default=os.path.join("figures", "ablation_group_summary.csv"))
    ap.add_argument("--out_md", default=os.path.join("figures", "ablation_summary.md"))
    args = ap.parse_args()

    rows = scan_runs(args.base_dir, args.pattern, args.eval_rel)
    if not rows:
        print("No runs found. Check --base_dir and --pattern.")
        return 0

    # Per-run CSV
    write_csv(
        args.out_csv,
        rows,
        header=["run_dir", "label_base", "seed", "macro_f1", "f1_healthy", "f1_degrading", "f1_fault"],
    )
    print(f"Wrote per-run CSV: {args.out_csv}")

    # Group summary
    groups = aggregate_groups(rows)
    write_csv(
        args.out_group_csv,
        groups,
        header=[
            "label_base", "n",
            "macro_f1_mean", "macro_f1_std",
            "f1_healthy_mean", "f1_healthy_std",
            "f1_degrading_mean", "f1_degrading_std",
            "f1_fault_mean", "f1_fault_std",
        ],
    )
    print(f"Wrote group summary CSV: {args.out_group_csv}")

    # Markdown summary
    write_markdown(args.out_md, groups)
    print(f"Wrote Markdown summary: {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

