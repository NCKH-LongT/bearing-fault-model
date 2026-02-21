import os
import sys
import argparse
import numpy as np
import torch
from typing import List, Dict, Tuple

# Ensure project root is on sys.path when running the script directly
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.logs_ttf import LogsTTFDataset
from features.spectrogram import SpectrogramTransform
from features.temp_features import temp_stats_window
from models.resnet2d import ResNet18Small

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def load_cfg(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_model(cfg: dict, ckpt_path: str, device):
    model = ResNet18Small(in_ch=2, num_classes=cfg["num_classes"], temp_feat_dim=6)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def build_eval_transform(cfg: dict):
    return SpectrogramTransform(
        n_fft=cfg["stft"]["n_fft"],
        hop_length=cfg["stft"]["hop_length"],
        window=cfg["stft"]["window"],
        log_add=cfg["stft"]["log_add"],
        target_size=tuple(cfg["input_size"]),
        training=False,
    )


@torch.no_grad()
def eval_split(ds: LogsTTFDataset, model, device, batch_size: int, max_windows: int = 50, verbose: bool = True) -> List[Dict]:
    out = []
    n_items = len(ds)
    for i in range(n_items):
        X, T, y = ds.get_all_windows(i)
        # Optionally subsample windows to speed up
        if isinstance(max_windows, int) and max_windows > 0 and X.shape[0] > max_windows:
            import numpy as _np
            idx = _np.linspace(0, X.shape[0] - 1, num=max_windows, dtype=int)
            X = X[idx]
            T = T[idx]
        y = int(y.item())
        logits_all = []
        for s in range(0, X.shape[0], batch_size):
            xb = X[s:s+batch_size].to(device)
            tb = T[s:s+batch_size].to(device)
            lb = model(xb, tb)
            logits_all.append(lb.cpu())
        logits = torch.cat(logits_all, dim=0)
        mean_logit = logits.mean(0)
        pred = int(mean_logit.argmax().item())
        prob = float(mean_logit.softmax(0).max().item())
        ttf = float(ds.items[i].get("ttf_percent", np.nan))
        out.append({"y": y, "pred": pred, "ttf": ttf, "conf": prob})
        if verbose and (i + 1) % 5 == 0:
            print(f"Evaluated {i+1}/{n_items} files...", flush=True)
    return out


def compute_bins(results: List[Dict], edges: List[float], labels: List[int]) -> Tuple[List[Tuple[float,float]], List[int], List[float], List[float]]:
    from sklearn.metrics import accuracy_score, f1_score
    bins = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    n, accs, f1s = [], [], []
    for lo, hi in bins:
        ys = [r["y"] for r in results if lo <= r["ttf"] < hi]
        ps = [r["pred"] for r in results if lo <= r["ttf"] < hi]
        if len(ys) == 0:
            n.append(0); accs.append(np.nan); f1s.append(np.nan); continue
        n.append(len(ys))
        try:
            accs.append(accuracy_score(ys, ps))
        except Exception:
            accs.append(np.nan)
        try:
            f1s.append(f1_score(ys, ps, average="macro", labels=labels, zero_division=0))
        except Exception:
            f1s.append(np.nan)
    return bins, n, accs, f1s


def class_hist_over_bins(results: List[Dict], edges: List[float], labels: List[int]) -> Tuple[np.ndarray, List[Tuple[float,float]]]:
    bins = [(edges[i], edges[i+1]) for i in range(len(edges)-1)]
    M = np.zeros((len(bins), len(labels)), dtype=int)
    for bi, (lo, hi) in enumerate(bins):
        ys = [r["y"] for r in results if lo <= r["ttf"] < hi]
        for y in ys:
            if y in labels:
                M[bi, labels.index(y)] += 1
    return M, bins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/logs_stft_full_temporal.yaml")
    ap.add_argument("--ckpt", default="runs/logs_stft/best.pt")
    ap.add_argument("--bins", default="0,70,80,90,95,97.5,100.1", help="Comma-separated TTF bin edges")
    ap.add_argument("--show", action="store_true", help="Display plots on screen")
    ap.add_argument("--max_windows", type=int, default=50, help="Max windows per file during eval (0 = all)")
    ap.add_argument("--out_subdir", default="eval/time_metrics", help="Subdirectory under log.out_dir to save outputs")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.ckpt, device)
    te_spec = build_eval_transform(cfg)

    split_mode = cfg.get("split_mode", "temporal")
    ttf_cfg = cfg.get("temporal_ttf", {"train": [0.0, 70.0], "val": [70.0, 80.0], "test": [80.0, 100.1]})

    def make_ds(split, ttf_split):
        return LogsTTFDataset(
            cfg["data_dir"], cfg["manifest"], split=split,
            sampling_rate=cfg["sampling_rate"],
            window_seconds=cfg["window_seconds"],
            hop_seconds=cfg["hop_seconds"],
            ttf_split=ttf_split,
            split_mode=split_mode,
            train_ratio=cfg.get("stratified", {}).get("train", 0.7),
            val_ratio=cfg.get("stratified", {}).get("val", 0.1),
            test_ratio=cfg.get("stratified", {}).get("test", 0.2),
            random_seed=cfg.get("random_seed", cfg["train"]["seed"]),
            transform=te_spec,
            temp_feature_fn=temp_stats_window,
            exclude_list=cfg.get("exclude_list"),
            limit_files=None,
            seconds_cap=cfg.get("debug", {}).get("seconds_cap"),
        )

    # Evaluate across train/val/test (so we cover the full 0-100% timeline)
    results: List[Dict] = []
    for split, key in [("train", "train"), ("val", "val"), ("test", "test")]:
        ds = make_ds(split, tuple(ttf_cfg.get(key, [0.0, 100.1])))
        if len(ds) == 0:
            continue
        print(f"Evaluating {split} split with {len(ds)} files...", flush=True)
        results.extend(eval_split(ds, model, device, batch_size=cfg["train"]["batch_size"], max_windows=args.max_windows, verbose=True))

    if not results:
        print("No results to plot (dataset empty?)")
        return

    labels_all = list(range(cfg["num_classes"]))
    edges = [float(x) for x in args.bins.split(",")]
    bins, n, accs, f1s = compute_bins(results, edges, labels_all)

    out_dir = os.path.join(cfg["log"]["out_dir"], args.out_subdir)
    os.makedirs(out_dir, exist_ok=True)
    # Save CSV
    import csv
    with open(os.path.join(out_dir, "time_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin_lo", "bin_hi", "n", "accuracy", "macro_f1"])
        for (lo, hi), nn, a, m in zip(bins, n, accs, f1s):
            w.writerow([lo, hi, nn, a, m])

    if plt is None:
        print("matplotlib not available; saved only CSV")
        return

    centers = [0.5*(lo+hi) for (lo, hi) in bins]

    # Line chart: accuracy and macro-F1 vs TTF
    fig = plt.figure(figsize=(6, 3.5))
    ax = fig.add_subplot(111)
    ax.plot(centers, accs, marker='o', label='Accuracy')
    ax.plot(centers, f1s, marker='s', label='Macro F1')
    ax.set_xlabel('TTF (%)')
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "time_metrics.png"), dpi=200)
    if args.show:
        try:
            plt.show(block=True)
        except TypeError:
            plt.show()
    else:
        plt.close(fig)

    # Stacked bar: class distribution across TTF bins (true labels)
    M, bins2 = class_hist_over_bins(results, edges, labels_all)
    if M.sum() > 0:
        fig2 = plt.figure(figsize=(6, 3.5))
        ax2 = fig2.add_subplot(111)
        bottom = np.zeros(M.shape[0])
        colors = ['#4daf4a', '#377eb8', '#e41a1c']
        cls_names = [None] * len(LogsTTFDataset.CLASS_MAP)
        for k, v in LogsTTFDataset.CLASS_MAP.items():
            cls_names[v] = k
        for ci in range(M.shape[1]):
            ax2.bar(range(M.shape[0]), M[:, ci], bottom=bottom, color=colors[ci % len(colors)], label=cls_names[ci])
            bottom += M[:, ci]
        ax2.set_xticks(range(M.shape[0]))
        ax2.set_xticklabels([f"{int(lo)}-{int(hi)}" for (lo, hi) in bins2])
        ax2.set_xlabel('TTF bins (%)')
        ax2.set_ylabel('#Files')
        ax2.legend()
        fig2.tight_layout()
        fig2.savefig(os.path.join(out_dir, "class_dist_over_ttf.png"), dpi=200)
        if args.show:
            plt.show()
        else:
            plt.close(fig2)


if __name__ == "__main__":
    main()
