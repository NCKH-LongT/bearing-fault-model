import os
import sys
import argparse
import subprocess
from typing import Dict, Any, Tuple


def run_cmd(cmd: list, cwd: str | None = None) -> int:
    print("$", " ".join(cmd))
    try:
        return subprocess.call(cmd, cwd=cwd)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Command failed to start: {e}")
        return 1


def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(cfg: Dict[str, Any], path: str):
    import yaml
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)


def parse_macro_f1(report_path: str) -> Tuple[float, Dict[str, float]]:
    if not os.path.exists(report_path):
        return 0.0, {}
    try:
        lines = [l.strip() for l in open(report_path, "r", encoding="utf-8").read().splitlines() if l.strip()]
    except Exception:
        return 0.0, {}
    per_cls: Dict[str, float] = {}
    macro = 0.0
    for ln in lines:
        low = ln.lower()
        if low.startswith("macro avg"):
            parts = [p for p in ln.split() if p]
            try:
                macro = float(parts[-2])
            except Exception:
                macro = 0.0
        else:
            toks = ln.split()
            if len(toks) == 5:
                name, _, _, f1, _ = toks
                try:
                    per_cls[name] = float(f1)
                except Exception:
                    pass
    return macro, per_cls


def quality_ok(macro_f1: float, per_cls: Dict[str, float], min_macro: float, min_each: float) -> bool:
    if macro_f1 < min_macro:
        return False
    for v in per_cls.values():
        if v < min_each:
            return False
    return True


def discover_existing_rounds(base_out_dir: str, prefix: str) -> int:
    try:
        names = os.listdir(base_out_dir)
    except Exception:
        return 0
    mx = 0
    for n in names:
        if n.startswith(prefix):
            try:
                k = int(n.replace(prefix, ""))
                mx = max(mx, k)
            except Exception:
                pass
    return mx


def propose_next_cfg(cfg: Dict[str, Any], round_idx: int, early_stopped: bool, resume_ckpt: str | None) -> Dict[str, Any]:
    import copy
    new_cfg = copy.deepcopy(cfg)
    tr = new_cfg.setdefault("train", {})
    opt = new_cfg.setdefault("optim", {})

    # Ensure safe defaults for temporal FT
    tr["early_stop"] = True
    tr["val_max_windows"] = tr.get("val_max_windows", 50) or 50
    tr["use_amp"] = True
    tr["balanced_sampling"] = True
    tr["use_class_weights"] = False
    opt["use_onecycle"] = False

    # Round-specific tweaks (do not increase epochs)
    if round_idx == 1:
        # baseline small LR for FT
        tr["lr"] = float(tr.get("lr", 3e-5))
        tr["early_stop_patience"] = max(int(tr.get("early_stop_patience", 10)), 12)
    elif round_idx == 2:
        # if stuck/early-stopped, nudge patience and stabilize val metric
        if early_stopped:
            tr["early_stop_patience"] = max(int(tr.get("early_stop_patience", 12)), 15)
        tr["val_max_windows"] = 0  # use all windows for a few epochs
    elif round_idx == 3:
        # small LR up/down to escape flat region
        tr["lr"] = max(1e-5, float(tr.get("lr", 3e-5)) * 1.25)
    elif round_idx == 4:
        # revert val windows to speed up
        tr["val_max_windows"] = 50
        # try a slight label smoothing bump
        tr["label_smoothing"] = float(tr.get("label_smoothing", 0.05)) + 0.02
    else:
        # last resort: reduce input complexity a bit
        st = new_cfg.setdefault("stft", {})
        st["n_fft"] = min(int(st.get("n_fft", 4096)), 2048)
        st["hop_length"] = min(int(st.get("hop_length", 1024)), 512)
        new_cfg["input_size"] = [160, 160]

    # Resume from previous best for FT if requested
    if resume_ckpt and os.path.exists(resume_ckpt):
        tr["init_from"] = resume_ckpt

    return new_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Base temporal YAML config path")
    ap.add_argument("--min_macro_f1", type=float, default=0.45)
    ap.add_argument("--min_class_f1", type=float, default=0.30)
    ap.add_argument("--max_rounds", type=int, default=6)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--continue", dest="do_continue", action="store_true", help="Continue from last FT round")
    ap.add_argument("--resume_from_prev", action="store_true", help="Init from previous round best.pt")
    args = ap.parse_args()

    try:
        base_cfg = load_yaml(args.config)
    except Exception as e:
        print(f"Failed to load config: {e}")
        sys.exit(1)

    base_out_dir = base_cfg.get("log", {}).get("out_dir", "runs/logs_stft_temporal")
    prefix = "auto_ft_r"

    # discover existing rounds
    start = 1
    if args.do_continue:
        exist = discover_existing_rounds(base_out_dir, prefix)
        if exist > 0:
            start = exist + 1
            print(f"Continue FT from round {start} (existing 1..{exist})")

    import copy
    cfg_r = copy.deepcopy(base_cfg)

    if start > 1:
        last_dir = os.path.join(base_out_dir, f"{prefix}{start-1}")
        last_cfg = os.path.join(last_dir, "config.yaml")
        if os.path.exists(last_cfg):
            try:
                cfg_r = load_yaml(last_cfg)
            except Exception:
                pass

    total = args.max_rounds
    for rnum in range(start, start + total):
        print(f"=== FT Round {rnum}/{start + total - 1} ===")

        round_out_dir = os.path.join(base_out_dir, f"{prefix}{rnum}")
        os.makedirs(round_out_dir, exist_ok=True)
        # set per-round out dir
        cfg_r.setdefault("log", {})["out_dir"] = round_out_dir

        # write config
        cfg_path = os.path.join(round_out_dir, "config.yaml")
        try:
            save_yaml(cfg_r, cfg_path)
        except Exception as e:
            print(f"Failed to save round config: {e}")
            continue

        # train
        rc = run_cmd([args.python, "train_logs.py", "--config", cfg_path])
        if rc != 0:
            print("Train failed for this round. Skipping to next.")
            # propose next config with early_stopped=True to adjust patience/val windows
            cfg_r = propose_next_cfg(cfg_r, rnum, early_stopped=True, resume_ckpt=None)
            continue

        # eval
        ckpt = os.path.join(round_out_dir, "best.pt")
        rc = run_cmd([args.python, "eval_logs.py", "--config", cfg_path, "--ckpt", ckpt])
        if rc != 0:
            print("Eval failed for this round. Skipping to next.")
            cfg_r = propose_next_cfg(cfg_r, rnum, early_stopped=False, resume_ckpt=ckpt if args.resume_from_prev else None)
            continue

        # metrics
        report_path = os.path.join(round_out_dir, "eval", "report.txt")
        macro, per_cls = parse_macro_f1(report_path)
        print(f"Round {rnum}: macro_f1={macro:.4f} | per-class={per_cls}")

        # detect early stop by comparing last epoch with configured epochs
        early_stopped = False
        try:
            import csv
            hist_csv = os.path.join(round_out_dir, "train_log.csv")
            last_epoch = None
            with open(hist_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    last_epoch = int(float(row.get("epoch", 0)))
            if last_epoch is not None:
                cfg_epochs = int(cfg_r.get("train", {}).get("epochs", last_epoch))
                early_stopped = last_epoch < cfg_epochs
        except Exception:
            early_stopped = False

        if quality_ok(macro, per_cls, args.min_macro_f1, args.min_class_f1):
            print(f"Stopping FT: reached target at round {rnum} (macro_f1={macro:.4f}).")
            print(f"Best FT config at: {cfg_path}")
            return

        # propose next config (keep epochs; adjust patience/lr/val windows); resume from this ckpt if requested
        cfg_r = propose_next_cfg(cfg_r, rnum, early_stopped=early_stopped, resume_ckpt=ckpt if args.resume_from_prev else None)

    print("Finished FT rounds. Consider revising temporal splits or LR if target not met.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

