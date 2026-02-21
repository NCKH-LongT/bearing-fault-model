import os
import sys
import time
import argparse
import subprocess
from typing import Tuple, Dict, Any

try:
    import yaml
except ImportError:
    print("PyYAML is required. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def run_cmd(cmd: list, cwd: str = None) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd, cwd=cwd)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def parse_macro_f1(report_path: str) -> Tuple[float, Dict[str, float]]:
    """Parse sklearn classification_report text to get macro F1 and per-class F1.
    Returns (macro_f1, {class_name: f1}). If file missing or parse fails, returns (0.0, {})."""
    if not os.path.exists(report_path):
        return 0.0, {}
    try:
        lines = [l.strip() for l in open(report_path, "r", encoding="utf-8").read().splitlines() if l.strip()]
    except Exception:
        return 0.0, {}

    per_class: Dict[str, float] = {}
    macro_f1 = 0.0
    for i, ln in enumerate(lines):
        # lines like: class  precision  recall  f1-score  support
        if ln.lower().startswith("macro avg"):
            parts = [p for p in ln.split() if p]
            # Expect format: macro avg <prec> <rec> <f1> <support>
            try:
                macro_f1 = float(parts[-2])
            except Exception:
                macro_f1 = 0.0
        else:
            # try parse class rows (name may be one token)
            toks = ln.split()
            if len(toks) == 5:
                name, _, _, f1, _ = toks
                try:
                    per_class[name] = float(f1)
                except Exception:
                    pass
    return macro_f1, per_class


def quality_ok(macro_f1: float, per_cls: Dict[str, float], min_macro: float, min_each: float) -> bool:
    if macro_f1 < min_macro:
        return False
    if per_cls:
        for k, v in per_cls.items():
            if v < min_each:
                return False
    return True


def propose_next_cfg(cfg: Dict[str, Any], round_idx: int, last_macro: float, early_stopped: bool, resume_ckpt: str | None) -> Dict[str, Any]:
    """Heuristic tweaks across rounds. Returns a shallow-cloned cfg with changes."""
    import copy
    new_cfg = copy.deepcopy(cfg)
    tr = new_cfg.setdefault("train", {})
    opt = new_cfg.setdefault("optim", {})

    # Ensure log section
    lg = new_cfg.setdefault("log", {})
    out_dir = lg.get("out_dir", "runs/auto")
    lg["out_dir"] = out_dir

    # Round-specific adjustments
    if round_idx == 0:
        # Baseline: balanced sampling on, turn off class weights to avoid double-counting
        tr["balanced_sampling"] = True
        tr["use_class_weights"] = False
        tr["early_stop"] = True
        tr["early_stop_patience"] = max(int(tr.get("early_stop_patience", 12)), 15)
        tr["val_max_windows"] = tr.get("val_max_windows", 50) or 50
        opt["use_onecycle"] = False
        tr["lr"] = float(tr.get("lr", 2e-4))
    elif round_idx == 1:
        # If vẫn kém, ưu tiên tăng patience (không tăng epochs) và tinh chỉnh LR
        if early_stopped:
            tr["early_stop_patience"] = max(int(tr.get("early_stop_patience", 15)), 20)
        # nhẹ nhàng tăng/giảm LR để thoát valley
        tr["lr"] = float(tr.get("lr", 2e-4)) * 1.25
    elif round_idx == 2:
        # Giữ epochs, thử giảm nhiễu val: dùng toàn bộ cửa sổ trong vài vòng
        tr["val_max_windows"] = 0
        # bật lại 50 ở vòng sau sẽ do script đặt lại nếu cần
    elif round_idx == 3:
        # Thử bật class weights nếu vẫn lệch
        tr["use_class_weights"] = True
    else:
        # Last resort: reduce input complexity một chút
        st = new_cfg.setdefault("stft", {})
        st["n_fft"] = min(int(st.get("n_fft", 4096)), 2048)
        st["hop_length"] = min(int(st.get("hop_length", 1024)), 512)
        new_cfg["input_size"] = [160, 160]

    # Nếu muốn resume từ vòng trước: nạp init_from và hạ LR nhẹ để fine-tune
    if resume_ckpt and os.path.exists(resume_ckpt):
        tr["init_from"] = resume_ckpt
        tr["lr"] = min(float(tr.get("lr", 2e-4)), 2.0e-4)

    return new_cfg


def discover_existing_rounds(base_out_dir: str) -> int:
    try:
        names = os.listdir(base_out_dir)
    except Exception:
        return 0
    mx = 0
    for n in names:
        if n.startswith("auto_r"):
            try:
                k = int(n.replace("auto_r", ""))
                mx = max(mx, k)
            except Exception:
                pass
    return mx


def try_read_macro(report_dir: str) -> float:
    rp = os.path.join(report_dir, "eval", "report.txt")
    m, _ = parse_macro_f1(rp)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base YAML config path")
    ap.add_argument("--min_macro_f1", type=float, default=0.4)
    ap.add_argument("--min_class_f1", type=float, default=0.2)
    ap.add_argument("--max_rounds", type=int, default=5, help="Total rounds target; resumes will run remaining rounds only")
    ap.add_argument("--python", default=sys.executable, help="Python executable to use")
    ap.add_argument("--resume_from_prev", action="store_true", help="Init from previous round's best.pt to fine-tune")
    ap.add_argument("--continue", dest="do_continue", action="store_true", help="Continue from last round if present, do not restart at round 1")
    args = ap.parse_args()

    base_cfg = load_yaml(args.config)
    base_out_dir = base_cfg.get("log", {}).get("out_dir", "runs/auto")

    # Determine starting round if continuing
    existing = discover_existing_rounds(base_out_dir) if args.do_continue else 0
    start_round = max(1, existing + 1)
    if existing > 0 and args.do_continue:
        print(f"Found existing rounds: 1..{existing}. Will continue from round {start_round}.")

    prev_cfg_path = args.config
    best_metric = -1.0
    best_round = -1

    # If continuing, load last round's config as starting point and seed best_metric from its eval
    if start_round > 1:
        last_dir = os.path.join(base_out_dir, f"auto_r{start_round-1}")
        last_cfg = os.path.join(last_dir, "config.yaml")
        if os.path.exists(last_cfg):
            cfg_r = load_yaml(last_cfg)
            try:
                best_metric = try_read_macro(last_dir)  # seed with last macro if available
                best_round = start_round - 1
            except Exception:
                pass
        else:
            import copy
            cfg_r = copy.deepcopy(base_cfg)
    else:
        import copy
        cfg_r = copy.deepcopy(base_cfg)

    total_rounds = args.max_rounds
    # If we already have some rounds and want total max_rounds, only run remaining
    remaining = max(0, total_rounds - (start_round - 1))
    if remaining == 0:
        print(f"Already reached max_rounds={total_rounds}. Nothing to do.")
        return

    for rnum in range(start_round, start_round + remaining):
        print(f"=== Round {rnum}/{start_round + remaining - 1} ===")

        # Per-round output directory to keep evals separated
        round_out_dir = os.path.join(base_out_dir, f"auto_r{rnum}")
        cfg_r.setdefault("log", {})["out_dir"] = round_out_dir

        # Write a derived config per round inside its own folder
        os.makedirs(round_out_dir, exist_ok=True)
        cfg_r_path = os.path.join(round_out_dir, "config.yaml")
        save_yaml(cfg_r, cfg_r_path)

        # Train
        rc = run_cmd([args.python, "train_logs.py", "--config", cfg_r_path])
        if rc != 0:
            print("Train failed, aborting.")
            sys.exit(rc)

        # Evaluate
        lg = cfg_r.get("log", {})
        od = lg.get("out_dir", round_out_dir)
        ckpt = os.path.join(od, "best.pt")
        rc = run_cmd([args.python, "eval_logs.py", "--config", cfg_r_path, "--ckpt", ckpt])
        if rc != 0:
            print("Eval failed, aborting.")
            sys.exit(rc)

        report_path = os.path.join(od, "eval", "report.txt")
        macro_f1, per_cls = parse_macro_f1(report_path)
        print(f"Round {rnum} macro_f1={macro_f1:.4f} | per-class={per_cls}")

        # Track best
        if macro_f1 > best_metric:
            best_metric = macro_f1
            best_round = rnum

        # Stop if good enough
        if quality_ok(macro_f1, per_cls, args.min_macro_f1, args.min_class_f1):
            print(f"Stopping: reached target at round {rnum} (macro_f1={macro_f1:.4f}).")
            print(f"Best config used: {cfg_r_path}")
            return

        # Prepare hints for next round
        # Detect early stop by comparing last logged epoch vs configured epochs
        try:
            import csv
            hist_csv = os.path.join(od, "train_log.csv")
            last_epoch = None
            with open(hist_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    last_epoch = int(float(row.get("epoch", 0)))
            early_stopped = False
            if last_epoch is not None:
                cfg_epochs = int(cfg_r.get("train", {}).get("epochs", last_epoch))
                early_stopped = last_epoch < cfg_epochs
        except Exception:
            early_stopped = False

        prev_ckpt = ckpt if args.resume_from_prev else None

        # Propose next config now (without increasing epochs)
        cfg_r = propose_next_cfg(cfg_r, rnum, best_metric, early_stopped, prev_ckpt)

    print(f"Finished {args.max_rounds} rounds. Best macro_f1={best_metric:.4f} at round {best_round}.")


if __name__ == "__main__":
    main()
