import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.logs_ttf import LogsTTFDataset
from features.spectrogram import SpectrogramTransform
from features.temp_features import resolve_temp_feature
from models.resnet2d import ResNet18Small

try:
    import matplotlib.pyplot as plt
except Exception:  # matplotlib may be missing; plotting will be skipped
    plt = None


def evaluate_filewise(ds, model, device, batch_size=32, agg="mean"):
    ys, ps = [], []
    with torch.no_grad():
        for i in range(len(ds)):
            X, T, y = ds.get_all_windows(i)
            y = int(y.item())
            outs = []
            n = X.shape[0]
            for s in range(0, n, batch_size):
                xb = X[s:s+batch_size].to(device)
                tb = T[s:s+batch_size].to(device)
                lb = model(xb, tb)
                outs.append(lb.cpu())
            logits = torch.cat(outs, dim=0)
            if agg == "mean":
                pred = int(logits.mean(0).argmax().item())
            else:
                pred = int(np.bincount(logits.argmax(1).numpy()).argmax())
            ys.append(y)
            ps.append(pred)
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(ys, ps, digits=4, zero_division=0))
    try:
        labels_all = list(range(len(LogsTTFDataset.CLASS_MAP)))
        print(confusion_matrix(ys, ps, labels=labels_all))
    except Exception:
        print(confusion_matrix(ys, ps))
    return ys, ps


def main(cfg_path: str, ckpt_path: str, show: bool = False, agg: str = "mean"):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        except Exception:
            pass
    else:
        print("Using CPU")

    te_spec = SpectrogramTransform(
        n_fft=cfg["stft"]["n_fft"],
        hop_length=cfg["stft"]["hop_length"],
        window=cfg["stft"]["window"],
        log_add=cfg["stft"]["log_add"],
        target_size=tuple(cfg["input_size"]),
        training=False,
    )

    split_mode = cfg.get("split_mode", "temporal")
    strat = cfg.get("stratified", {"train": 0.7, "val": 0.1, "test": 0.2})
    rnd_seed = cfg.get("random_seed", cfg["train"]["seed"])
    # Optional temporal split ranges from config
    split_mode = cfg.get("split_mode", "temporal")
    ttf_cfg = cfg.get("temporal_ttf", {}) if split_mode == "temporal" else {}
    ttf_test = tuple(ttf_cfg.get("test", (80.0, 100.1))) if split_mode == "temporal" else (0.8, 1.0)

    model_cfg = cfg.get("model", {}) or {}
    use_temp = bool(model_cfg.get("use_temp", True))
    temp_feat_fn = None
    temp_feat_dim = 0
    temp_ctx_seconds = None
    temp_ctx_causal = True
    if use_temp:
        temp_cfg = model_cfg.get("temp_feature", {}) or {}
        temp_type = temp_cfg.get("type", "stats6")
        temp_feat_fn, temp_feat_dim = resolve_temp_feature(temp_type)
        temp_ctx_seconds = temp_cfg.get("context_seconds")
        temp_ctx_seconds = float(temp_ctx_seconds) if temp_ctx_seconds is not None else None
        temp_ctx_causal = bool(temp_cfg.get("causal", True))

    test_ds = LogsTTFDataset(
        cfg["data_dir"], cfg["manifest"], split="test",
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        ttf_split=ttf_test,
        split_mode=split_mode,
        train_ratio=strat.get("train", 0.7),
        val_ratio=strat.get("val", 0.1),
        test_ratio=strat.get("test", 0.2),
        min_per_class_val=strat.get("min_per_class_val"),
        min_per_class_test=strat.get("min_per_class_test"),
        random_seed=rnd_seed,
        transform=te_spec,
        temp_feature_fn=temp_feat_fn,
        temp_feat_dim=temp_feat_dim,
        temp_context_seconds=temp_ctx_seconds,
        temp_context_causal=temp_ctx_causal,
        exclude_list=cfg.get("exclude_list"),
        limit_files=cfg.get("debug", {}).get("limit_files_test"),
        seconds_cap=cfg.get("debug", {}).get("seconds_cap"),
    )
    pin_mem = torch.cuda.is_available()
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        pin_memory=pin_mem,
    )

    model = ResNet18Small(in_ch=2, num_classes=cfg["num_classes"], temp_feat_dim=temp_feat_dim)
    # Safe, forward-compatible load: prefer weights_only and accept both formats
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state
    model.load_state_dict(sd)
    model.to(device)

    print(f"Test set report (file-wise, {agg}-agg):")
    ys, ps = evaluate_filewise(test_ds, model, device, batch_size=cfg["train"]["batch_size"], agg=agg)

    # Prepare output directory
    out_dir = os.path.join(cfg["log"]["out_dir"], "eval")
    os.makedirs(out_dir, exist_ok=True)

    # Save textual report and confusion matrix values
    from sklearn.metrics import classification_report, confusion_matrix, f1_score
    cls_names = [None] * len(LogsTTFDataset.CLASS_MAP)
    for k, v in LogsTTFDataset.CLASS_MAP.items():
        cls_names[v] = k
    labels_all = list(range(len(cls_names)))

    # Always produce a full 3-class report (healthy/degrading/fault),
    # using zero_division=0 for any missing classes.
    report_full = classification_report(
        ys, ps,
        labels=labels_all,
        target_names=cls_names,
        digits=4,
        zero_division=0,
    )
    with open(os.path.join(out_dir, "report.txt"), "w", encoding="utf-8") as f:
        f.write(report_full + "\n")

    # Additionally, keep a present-labels-only report for reference
    try:
        labels_present = sorted(set(ys) | set(ps))
        if labels_present:
            names_present = [cls_names[i] for i in labels_present]
            report_present = classification_report(
                ys, ps,
                labels=labels_present,
                target_names=names_present,
                digits=4,
                zero_division=0,
            )
            with open(os.path.join(out_dir, "report_present.txt"), "w", encoding="utf-8") as f:
                f.write(report_present + "\n")
    except Exception:
        pass
    # Confusion matrix with full label set for consistent axes
    cm = confusion_matrix(ys, ps, labels=labels_all)
    np.savetxt(os.path.join(out_dir, "confusion_matrix.csv"), cm, fmt="%d", delimiter=",")

    # Plot confusion matrix and per-class F1 if matplotlib is available
    if plt is not None:
        try:
            fig_cm = plt.figure(figsize=(4, 4))
            ax = fig_cm.add_subplot(111)
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(cls_names)))
            ax.set_yticks(range(len(cls_names)))
            ax.set_xticklabels(cls_names, rotation=45, ha="right")
            ax.set_yticklabels(cls_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
            fig_cm.tight_layout()
            fig_cm.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=200)
            if show:
                plt.show()
            else:
                plt.close(fig_cm)

            f1_per_class = f1_score(ys, ps, average=None, labels=labels_all, zero_division=0)
            fig_f1 = plt.figure(figsize=(5, 3))
            ax2 = fig_f1.add_subplot(111)
            ax2.bar(range(len(cls_names)), f1_per_class)
            ax2.set_xticks(range(len(cls_names)))
            ax2.set_xticklabels(cls_names, rotation=45, ha="right")
            ax2.set_ylim(0, 1)
            ax2.set_ylabel("F1-score")
            fig_f1.tight_layout()
            fig_f1.savefig(os.path.join(out_dir, "f1_per_class.png"), dpi=200)
            if show:
                plt.show()
            else:
                plt.close(fig_f1)

            # Additionally, plot present-classes-only F1 (omit absent classes)
            try:
                labels_present = sorted(set(ys) | set(ps))
                if labels_present:
                    names_present = [cls_names[i] for i in labels_present]
                    f1_present = f1_score(ys, ps, average=None, labels=labels_present, zero_division=0)
                    fig_f1p = plt.figure(figsize=(4.5, 3))
                    axp = fig_f1p.add_subplot(111)
                    axp.bar(range(len(names_present)), f1_present)
                    axp.set_xticks(range(len(names_present)))
                    axp.set_xticklabels(names_present, rotation=45, ha="right")
                    axp.set_ylim(0, 1)
                    axp.set_ylabel("F1-score (present)")
                    fig_f1p.tight_layout()
                    fig_f1p.savefig(os.path.join(out_dir, "f1_present.png"), dpi=200)
                    if show:
                        plt.show()
                    else:
                        plt.close(fig_f1p)
            except Exception:
                pass
        except Exception:
            pass

    # Early vs Late TTF subsets (if metadata available)
    try:
        ttfs = [float(test_ds.items[i].get("ttf_percent", np.nan)) for i in range(len(test_ds))]
        ttfs = np.array(ttfs, dtype=float)
        ys_a, ps_a = [y for (y, t) in zip(ys, ttfs) if 70.0 <= t <= 90.0], [p for (p, t) in zip(ps, ttfs) if 70.0 <= t <= 90.0]
        ys_b, ps_b = [y for (y, t) in zip(ys, ttfs) if 90.0 < t <= 100.1], [p for (p, t) in zip(ps, ttfs) if 90.0 < t <= 100.1]
        if ys_a:
            labels_present_a = sorted(set(ys_a) | set(ps_a))
            names_present_a = [cls_names[i] for i in labels_present_a]
            rep_a = classification_report(ys_a, ps_a, labels=labels_present_a, target_names=names_present_a, digits=4, zero_division=0)
            with open(os.path.join(out_dir, "report_early_70_90.txt"), "w", encoding="utf-8") as f:
                f.write(rep_a + "\n")
            cma = confusion_matrix(ys_a, ps_a, labels=labels_all)
            np.savetxt(os.path.join(out_dir, "confusion_matrix_early.csv"), cma, fmt="%d", delimiter=",")
            if plt is not None:
                try:
                    fig = plt.figure(figsize=(4, 4))
                    ax = fig.add_subplot(111)
                    ax.imshow(cma, cmap="Blues")
                    ax.set_xticks(range(len(cls_names)))
                    ax.set_yticks(range(len(cls_names)))
                    ax.set_xticklabels(cls_names, rotation=45, ha="right")
                    ax.set_yticklabels(cls_names)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    for i in range(cma.shape[0]):
                        for j in range(cma.shape[1]):
                            ax.text(j, i, str(cma[i, j]), ha="center", va="center", color="black")
                    fig.tight_layout()
                    fig.savefig(os.path.join(out_dir, "confusion_matrix_early.png"), dpi=200)
                    if show:
                        plt.show()
                    else:
                        plt.close(fig)
                except Exception:
                    pass
        if ys_b:
            labels_present_b = sorted(set(ys_b) | set(ps_b))
            names_present_b = [cls_names[i] for i in labels_present_b]
            rep_b = classification_report(ys_b, ps_b, labels=labels_present_b, target_names=names_present_b, digits=4, zero_division=0)
            with open(os.path.join(out_dir, "report_late_90_100.txt"), "w", encoding="utf-8") as f:
                f.write(rep_b + "\n")
            cmb = confusion_matrix(ys_b, ps_b, labels=labels_all)
            np.savetxt(os.path.join(out_dir, "confusion_matrix_late.csv"), cmb, fmt="%d", delimiter=",")
            if plt is not None:
                try:
                    fig = plt.figure(figsize=(4, 4))
                    ax = fig.add_subplot(111)
                    ax.imshow(cmb, cmap="Blues")
                    ax.set_xticks(range(len(cls_names)))
                    ax.set_yticks(range(len(cls_names)))
                    ax.set_xticklabels(cls_names, rotation=45, ha="right")
                    ax.set_yticklabels(cls_names)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("True")
                    for i in range(cmb.shape[0]):
                        for j in range(cmb.shape[1]):
                            ax.text(j, i, str(cmb[i, j]), ha="center", va="center", color="black")
                    fig.tight_layout()
                    fig.savefig(os.path.join(out_dir, "confusion_matrix_late.png"), dpi=200)
                    if show:
                        plt.show()
                    else:
                        plt.close(fig)
                except Exception:
                    pass
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/logs_stft.yaml")
    ap.add_argument("--ckpt", default="runs/logs_stft/best.pt")
    ap.add_argument("--show", action="store_true", help="Display eval plots on screen")
    ap.add_argument("--agg", default="mean", choices=["mean", "vote"], help="File-wise aggregation: mean (pre-softmax logits) or majority vote")
    args = ap.parse_args()
    import os as _os
    env_cfg = _os.environ.get("BF_CONFIG")
    cfg_path = env_cfg if env_cfg else args.config
    print(f"Config: {cfg_path}")
    main(cfg_path, args.ckpt, show=args.show, agg=args.agg)
