import os
import sys
import argparse
import itertools
import subprocess
from typing import Dict, Any, Tuple


def run_cmd(cmd: list) -> int:
    print("$", " ".join(cmd))
    try:
        return subprocess.call(cmd)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Failed to run command: {e}")
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


def parse_class_f1(report_path: str, class_name: str) -> float:
    """Extract f1-score for a given class from sklearn classification_report text.
    Returns 0.0 if not found or parse fails.
    """
    if not os.path.exists(report_path):
        return 0.0
    try:
        lines = [l.strip() for l in open(report_path, "r", encoding="utf-8").read().splitlines() if l.strip()]
    except Exception:
        return 0.0
    for ln in lines:
        toks = ln.split()
        # expected row: <name> <prec> <recall> <f1> <support>
        if len(toks) == 5 and toks[0].lower() == class_name.lower():
            try:
                return float(toks[3])
            except Exception:
                return 0.0
    return 0.0


def objective(score_deg: float, score_fault: float, min_fault: float) -> float:
    """Composite objective prioritizing Degrading F1 in [70,90]% while keeping Fault F1 >= min_fault in [90,100.1]%.
    Returns a negative penalty if Fault is below threshold to discourage such configs.
    """
    if score_fault < min_fault:
        # heavy penalty if the fault F1 is under threshold
        return score_deg - (min_fault - score_fault) * 2.0
    # weighted sum (tweakable): emphasize Degrading but keep some weight on Fault
    return 0.7 * score_deg + 0.3 * score_fault


def main():
    ap = argparse.ArgumentParser(description="Auto search configs to improve Degrading F1 on [70,90]% while preserving Fault F1 on [90,100.1]%")
    ap.add_argument("--base_config", required=True, help="Path to a base temporal YAML config")
    ap.add_argument("--python", default=sys.executable, help="Python executable to run train/eval")
    ap.add_argument("--min_fault_f1", type=float, default=0.55, help="Minimum acceptable Fault F1 on [90,100.1]%")
    ap.add_argument("--output_root", default="runs/auto_temp_gating", help="Root dir to place derived runs")
    ap.add_argument("--max_trials", type=int, default=12, help="Limit total trials")
    args = ap.parse_args()

    base = load_yaml(args.base_config)
    os.makedirs(args.output_root, exist_ok=True)

    # Search space: designed to work with current code (gating flags are passed through for compatibility)
    res_opts = [
        {"stft": {"n_fft": 2048, "hop_length": 512}, "input_size": [160, 160]},
        {"stft": {"n_fft": 4096, "hop_length": 1024}, "input_size": [224, 224]},
    ]
    lr_opts = [float(base.get("train", {}).get("lr", 2e-4)), float(base.get("train", {}).get("lr", 2e-4)) * 0.75]
    cw_opts = [False, True]
    ls_opts = [0.05, 0.0]
    md_opts = [0.0, 0.3]  # modality dropout p (used if gating is implemented)
    gating_opts = [False, True]

    trial = 0
    best = {"score": -1e9, "dir": None, "config": None, "deg": 0.0, "fault": 0.0}

    for (res, lr, cw, ls, md, gt) in itertools.product(res_opts, lr_opts, cw_opts, ls_opts, md_opts, gating_opts):
        if trial >= args.max_trials:
            break
        trial += 1

        # clone base
        import copy
        cfg = copy.deepcopy(base)
        # update resolution
        st = cfg.setdefault("stft", {})
        st.update(res["stft"])  # n_fft, hop_length
        cfg["input_size"] = res["input_size"]
        # training knobs
        tr = cfg.setdefault("train", {})
        tr["lr"] = lr
        tr["use_class_weights"] = cw
        tr["label_smoothing"] = ls
        tr["early_stop"] = True
        tr.setdefault("early_stop_patience", 12)
        # fusion/temperature flags (compatible even if model ignores them)
        cfg.setdefault("features", {}).setdefault("temp", {})
        cfg.setdefault("fusion", {})["gating"] = gt
        cfg["fusion"]["modality_dropout_p"] = md
        # eval slices hint for documentation/debugging
        cfg.setdefault("eval", {})["slices"] = [[70, 90.0], [90.0, 100.1], [70.0, 100.1]]

        # set per-trial output dir
        out_dir = os.path.join(args.output_root, f"trial_{trial:02d}")
        cfg.setdefault("log", {})["out_dir"] = out_dir

        # save config
        cfg_path = os.path.join(out_dir, "config.yaml")
        save_yaml(cfg, cfg_path)

        # train
        rc = run_cmd([args.python, "train_logs.py", "--config", cfg_path])
        if rc != 0:
            print(f"Trial {trial}: train failed; skipping")
            continue

        # eval (uses dataset temporal split from cfg)
        ckpt = os.path.join(out_dir, "best.pt")
        rc = run_cmd([args.python, "eval_logs.py", "--config", cfg_path, "--ckpt", ckpt])
        if rc != 0:
            print(f"Trial {trial}: eval failed; skipping")
            continue

        # parse metrics for the two slices
        eval_dir = os.path.join(out_dir, "eval")
        rep_early = os.path.join(eval_dir, "report_early_70_90.txt")
        rep_late = os.path.join(eval_dir, "report_late_90_100.txt")
        f1_deg = parse_class_f1(rep_early, "degrading")
        f1_fault = parse_class_f1(rep_late, "fault")
        score = objective(f1_deg, f1_fault, min_fault=args.min_fault_f1)
        print(f"Trial {trial}: F1_deg[70,90]={f1_deg:.4f}, F1_fault[90,100.1]={f1_fault:.4f}, score={score:.4f}")

        if score > best["score"]:
            best.update({"score": score, "dir": out_dir, "config": cfg_path, "deg": f1_deg, "fault": f1_fault})

    if best["dir"] is None:
        print("No successful trials. Please check data/config and try again.")
        sys.exit(2)

    print("=== Best trial ===")
    print(f"dir       : {best['dir']}")
    print(f"config    : {best['config']}")
    print(f"F1_deg[70,90]   : {best['deg']:.4f}")
    print(f"F1_fault[90,100.1]: {best['fault']:.4f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

