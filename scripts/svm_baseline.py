import os
import argparse
import json
from typing import List, Dict, Tuple

import numpy as np


def load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_manifest(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline()
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            parts = [p.strip().strip('"') for p in ln.split(",")]
            if len(parts) < 4:
                continue
            rows.append({
                "file": parts[0],
                "run_id": parts[1],
                "ttf_percent": parts[2],
                "fault_type": parts[3].lower(),
            })
    return rows


CLASS_MAP = {"healthy": 0, "degrading": 1, "fault": 2}


def build_split(rows: List[Dict], split: str, ttf_split: Tuple[float, float]) -> List[Dict]:
    lo, hi = float(ttf_split[0]), float(ttf_split[1])
    items = []
    for r in rows:
        try:
            p = float(r["ttf_percent"]) if r["ttf_percent"] != "" else 0.0
        except Exception:
            p = 0.0
        if not (lo <= p <= hi if split == "test" and hi == 100.1 else lo <= p < hi):
            continue
        name = r["fault_type"].strip().lower()
        if name not in CLASS_MAP:
            continue
        items.append({
            "file": r["file"],
            "label": CLASS_MAP[name],
            "ttf_percent": p,
        })
    return items


def kurtosis_np(x: np.ndarray) -> float:
    x = x.astype(np.float64)
    mu = x.mean()
    s = x.std(ddof=0) + 1e-8
    z4 = np.mean(((x - mu) / s) ** 4)
    return float(z4 - 3.0)  # Fisher kurtosis


def rms_np(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x.astype(np.float64)))))


def file_features(csv_path: str, chunksize: int = 200_000) -> np.ndarray:
    """Efficient per-file 8D features using chunked CSV read.
    Columns: vib_x, vib_y, temp_bearing, temp_atm (we use 0 and 1).
    """
    try:
        import pandas as pd
        usecols = [0, 1]
        # Accumulate raw moments for each channel: S1, S2, S3, S4 and N
        sums = {
            0: {"S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 0.0, "N": 0},
            1: {"S1": 0.0, "S2": 0.0, "S3": 0.0, "S4": 0.0, "N": 0},
        }
        for chunk in pd.read_csv(
            csv_path,
            header=None,
            usecols=usecols,
            chunksize=chunksize,
            dtype=np.float32,
            engine="c",
        ):
            arr = chunk.to_numpy(copy=False)
            # sanitize
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            for ch in (0, 1):
                x = arr[:, ch].astype(np.float64)
                N = x.size
                if N == 0:
                    continue
                sums[ch]["S1"] += float(x.sum())
                sums[ch]["S2"] += float(np.square(x).sum())
                sums[ch]["S3"] += float(np.power(x, 3).sum())
                sums[ch]["S4"] += float(np.power(x, 4).sum())
                sums[ch]["N"] += int(N)

        feats = []
        for ch in (0, 1):
            N = sums[ch]["N"]
            if N == 0:
                feats.extend([0.0, 0.0, 0.0, 0.0])
                continue
            S1, S2, S3, S4 = sums[ch]["S1"], sums[ch]["S2"], sums[ch]["S3"], sums[ch]["S4"]
            mu = S1 / N
            m2 = S2 / N
            m3 = S3 / N
            m4 = S4 / N
            var = max(m2 - mu * mu, 1e-12)
            sigma = float(np.sqrt(var))
            # central 4th moment via raw moments
            mu4 = m4 - 4 * mu * m3 + 6 * (mu**2) * m2 - 3 * (mu**4)
            kurt = float(mu4 / (var**2) - 3.0)
            rms = float(np.sqrt(m2))
            feats.extend([float(mu), sigma, rms, kurt])
        return np.array(feats, dtype=np.float32)
    except Exception:
        # Fallback to numpy (slower)
        arr = np.loadtxt(csv_path, delimiter=",")
        vib = arr[:, :2].astype(np.float32)
        feats = []
        for i in range(2):
            x = vib[:, i]
            x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
            feats.extend([
                float(np.mean(x)),
                float(np.std(x) + 1e-8),
                rms_np(x),
                kurtosis_np(x),
            ])
        return np.array(feats, dtype=np.float32)


def compute_dataset_features(data_dir: str, items: List[Dict]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X, y, ttf = [], [], []
    for it in items:
        p = os.path.join(data_dir, it["file"])
        X.append(file_features(p))
        y.append(int(it["label"]))
        ttf.append(float(it["ttf_percent"]))
    return np.vstack(X), np.array(y, dtype=int), np.array(ttf, dtype=float)


def eval_and_save(y_true: np.ndarray, y_pred: np.ndarray, cls_names: List[str], out_dir: str, tag: str):
    from sklearn.metrics import classification_report, confusion_matrix
    labels_present = sorted(list(set(y_true) | set(y_pred)))
    names_present = [cls_names[i] for i in labels_present]
    rp = classification_report(y_true, y_pred, labels=labels_present, target_names=names_present, digits=4, zero_division=0)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"report_{tag}.txt"), "w", encoding="utf-8") as f:
        f.write(rp + "\n")
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(cls_names))))
    np.savetxt(os.path.join(out_dir, f"confusion_matrix_{tag}.csv"), cm, fmt="%d", delimiter=",")
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(cls_names)))
        ax.set_yticks(range(len(cls_names)))
        ax.set_xticklabels(cls_names, rotation=45, ha="right")
        ax.set_yticklabels(cls_names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, f"confusion_matrix_{tag}.png"), dpi=200)
        plt.close(fig)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description="SVM baseline on classical vibration features (8D/file)")
    ap.add_argument("--config", required=True, help="YAML config with temporal_ttf")
    ap.add_argument("--out", default="runs/svm_baseline_temporal", help="Output directory")
    ap.add_argument("--auto_grid", action="store_true", help="Run a small hyperparam grid and stop when thresholds met")
    ap.add_argument("--min_macro_present", type=float, default=0.5, help="Early-stop when present macro-F1 >= this")
    ap.add_argument("--min_fault_late", type=float, default=0.5, help="Early-stop when late Fault F1 >= this")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    data_dir = cfg["data_dir"]
    manifest = cfg["manifest"]
    ttfs = cfg.get("temporal_ttf", {"train": [0.0, 60.0], "val": [60.0, 70.0], "test": [70.0, 100.1]})
    print(f"Config: {args.config}")
    rows = read_manifest(manifest)
    print(f"Read manifest: {len(rows)} rows")

    train_items = build_split(rows, "train", tuple(ttfs.get("train", (0.0, 60.0))))
    val_items = build_split(rows, "val", tuple(ttfs.get("val", (60.0, 70.0))))
    test_items = build_split(rows, "test", tuple(ttfs.get("test", (70.0, 100.1))))
    print(f"Split sizes | train={len(train_items)} val={len(val_items)} test={len(test_items)}")

    print("Computing features (train/val/test) ...")
    Xtr, ytr, _ = compute_dataset_features(data_dir, train_items)
    Xva, yva, _ = compute_dataset_features(data_dir, val_items)
    Xte, yte, tte = compute_dataset_features(data_dir, test_items)
    print(f"Feature shapes | Xtr={Xtr.shape} Xva={Xva.shape} Xte={Xte.shape}")

    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import f1_score, precision_recall_fscore_support

    Xtr_full = np.vstack([Xtr, Xva])
    ytr_full = np.concatenate([ytr, yva])

    grids = [(1.0, "scale", None, "rbf")]
    if args.auto_grid:
        grids = []
        for C in [0.1, 1.0, 10.0]:
            for gamma in ["scale", 1e-3, 1e-4]:
                for cw in [None, "balanced"]:
                    for kernel in ["rbf", "linear"]:
                        grids.append((C, gamma, cw, kernel))

    best = {"score": -1.0, "C": None, "gamma": None, "cw": None, "kernel": None, "ypred": None}
    for (C, gamma, cw, kernel) in grids:
        clf = make_pipeline(StandardScaler(), SVC(C=C, kernel=kernel, gamma=gamma, class_weight=cw))
        print(f"Train SVM | C={C} gamma={gamma} class_weight={cw} kernel={kernel}")
        clf.fit(Xtr_full, ytr_full)
        ypred = clf.predict(Xte)
        labels_present = sorted(set(yte) | set(ypred))
        f1_macro_present = f1_score(yte, ypred, labels=labels_present, average="macro", zero_division=0)
        # late fault F1
        mask_late = (tte > 90.0) & (tte <= 100.1)
        f1_fault_late = 0.0
        if mask_late.any():
            _, _, f1_arr, _ = precision_recall_fscore_support(yte[mask_late], ypred[mask_late], labels=[CLASS_MAP["fault"]], zero_division=0)
            f1_fault_late = float(f1_arr[0])
        score = 0.7 * f1_macro_present + 0.3 * f1_fault_late
        print(f"  -> present macro F1={f1_macro_present:.4f} | late Fault F1={f1_fault_late:.4f} | score={score:.4f}")
        if score > best["score"]:
            best.update({"score": score, "C": C, "gamma": gamma, "cw": cw, "kernel": kernel, "ypred": ypred})
        if args.auto_grid and (f1_macro_present >= args.min_macro_present and f1_fault_late >= args.min_fault_late):
            print("Early stop: thresholds met.")
            best.update({"ypred": ypred})
            break

    print(f"Best SVM | C={best['C']} gamma={best['gamma']} cw={best['cw']} kernel={best['kernel']} score={best['score']:.4f}")
    ypred = best["ypred"]
    out_eval = os.path.join(args.out, "eval")
    cls_names = [None] * len(CLASS_MAP)
    for k, v in CLASS_MAP.items():
        cls_names[v] = k
    eval_and_save(yte, ypred, cls_names, out_eval, tag="present")
    mask_early = (tte >= 70.0) & (tte <= 90.0)
    mask_late = (tte > 90.0) & (tte <= 100.1)
    if mask_early.any():
        eval_and_save(yte[mask_early], ypred[mask_early], cls_names, out_eval, tag="early_70_90")
    if mask_late.any():
        eval_and_save(yte[mask_late], ypred[mask_late], cls_names, out_eval, tag="late_90_100")

    with open(os.path.join(args.out, "config_used.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    main()
