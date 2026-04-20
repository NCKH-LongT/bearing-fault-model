from __future__ import annotations

import csv
import os
import pickle
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from datasets.logs_ttf import LogsTTFDataset

from classical_baselines.features import resolve_feature_extractor


CLASS_NAMES = [None] * len(LogsTTFDataset.CLASS_MAP)
for _name, _idx in LogsTTFDataset.CLASS_MAP.items():
    CLASS_NAMES[_idx] = _name


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_split_items(cfg: dict, split: str, override_split_mode: str = None) -> List[Dict]:
    # allow caller to override split_mode (e.g. train on stratified, eval on temporal)
    split_mode = override_split_mode or cfg.get("split_mode", "temporal")
    strat = cfg.get("stratified", {}) or {}
    rnd_seed = cfg.get("random_seed", 42)
    debug_cfg = cfg.get("debug", {}) or {}

    if split_mode == "temporal":
        ttf_cfg = cfg.get("temporal_ttf", {}) or {}
        fallback = {
            "train": (0.0, 60.0),
            "val": (60.0, 70.0),
            "test": (70.0, 100.0),
        }
        ttf_split = tuple(ttf_cfg.get(split, fallback[split]))
    else:
        ttf_split = (0.0, 100.0)

    ds = LogsTTFDataset(
        cfg["data_dir"],
        cfg["manifest"],
        split=split,
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        split_mode=split_mode,
        ttf_split=ttf_split,
        train_ratio=strat.get("train", 0.6),
        val_ratio=strat.get("val", 0.2),
        test_ratio=strat.get("test", 0.2),
        min_per_class_val=strat.get("min_per_class_val"),
        min_per_class_test=strat.get("min_per_class_test"),
        random_seed=rnd_seed,
        transform=None,
        temp_feature_fn=None,
        temp_feat_dim=0,
        exclude_list=cfg.get("exclude_list"),
        limit_files=debug_cfg.get(f"limit_files_{split}"),
        seconds_cap=debug_cfg.get("seconds_cap"),
    )
    return ds.items


def make_windows(n: int, win: int, hop: int) -> List[Tuple[int, int]]:
    if n < win:
        return []
    out = []
    pos = 0
    while pos + win <= n:
        out.append((pos, pos + win))
        pos += hop
    return out


def read_signal_csv(path: str, max_rows: Optional[int] = None) -> np.ndarray:
    kwargs = {"delimiter": ","}
    if max_rows is not None and max_rows > 0:
        kwargs["max_rows"] = int(max_rows)
    arr = np.loadtxt(path, **kwargs)
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected at least 2 signal columns in {path}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def extract_window_features(
    item: Dict,
    win: int,
    hop: int,
    feature_name: str,
    seconds_cap: Optional[float],
    sampling_rate: int,
) -> np.ndarray:
    cap = int(seconds_cap * sampling_rate) if seconds_cap else None
    arr = read_signal_csv(item["path"], max_rows=cap)
    vib = arr[:, :2]
    extractor = resolve_feature_extractor(feature_name)
    feats = []
    for s, e in make_windows(vib.shape[0], win, hop):
        feats.append(extractor(vib[s:e]))
    if not feats:
        return np.zeros((0, 8), dtype=np.float32)
    return np.stack(feats, axis=0).astype(np.float32)


def build_training_matrix(cfg: dict, split: str, override_split_mode: str = None) -> Tuple[np.ndarray, np.ndarray]:
    items = build_split_items(cfg, split, override_split_mode=override_split_mode)
    win = int(round(float(cfg["window_seconds"]) * int(cfg["sampling_rate"])))
    hop = int(round(float(cfg["hop_seconds"]) * int(cfg["sampling_rate"])))
    feature_name = cfg["classical"]["feature_name"]
    seconds_cap = (cfg.get("debug", {}) or {}).get("seconds_cap")
    sampling_rate = int(cfg["sampling_rate"])

    x_all: List[np.ndarray] = []
    y_all: List[int] = []
    for item in items:
        feats = extract_window_features(item, win, hop, feature_name, seconds_cap, sampling_rate)
        if feats.size == 0:
            continue
        x_all.append(feats)
        y_all.extend([int(item["label"])] * feats.shape[0])
    if not x_all:
        raise RuntimeError(f"No windows extracted for split={split}.")
    X = np.concatenate(x_all, axis=0)
    y = np.asarray(y_all, dtype=np.int64)
    return X, y


def build_model(cfg: dict):
    model_type = (cfg["classical"]["model_type"] or "svm").strip().lower()
    seed = int(cfg.get("random_seed", 42))

    if model_type == "svm":
        params = cfg["classical"].get("svm", {}) or {}
        clf = SVC(
            C=float(params.get("C", 1.0)),
            kernel=params.get("kernel", "rbf"),
            gamma=params.get("gamma", "scale"),
            class_weight=params.get("class_weight", "balanced"),
            probability=True,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type == "logreg":
        params = cfg["classical"].get("logreg", {}) or {}
        clf = LogisticRegression(
            C=float(params.get("C", 1.0)),
            max_iter=int(params.get("max_iter", 2000)),
            class_weight=params.get("class_weight", "balanced"),
            multi_class="auto",
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", clf)])

    if model_type == "rf":
        params = cfg["classical"].get("rf", {}) or {}
        clf = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 300)),
            max_depth=params.get("max_depth"),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            class_weight=params.get("class_weight", "balanced"),
            random_state=seed,
            n_jobs=-1,
        )
        return Pipeline([("clf", clf)])

    raise ValueError(f"Unsupported classical model_type: {model_type}")


def _window_scores(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.asarray(scores, dtype=np.float32)
        if scores.ndim == 1:
            scores = np.stack([-scores, scores], axis=1)
        return scores
    pred = model.predict(X)
    scores = np.zeros((len(pred), len(CLASS_NAMES)), dtype=np.float32)
    for i, p in enumerate(pred):
        scores[i, int(p)] = 1.0
    return scores


def evaluate_filewise(cfg: dict, model, split: str, override_split_mode: str = None) -> Tuple[List[int], List[int]]:
    items = build_split_items(cfg, split, override_split_mode=override_split_mode)
    win = int(round(float(cfg["window_seconds"]) * int(cfg["sampling_rate"])))
    hop = int(round(float(cfg["hop_seconds"]) * int(cfg["sampling_rate"])))
    feature_name = cfg["classical"]["feature_name"]
    seconds_cap = (cfg.get("debug", {}) or {}).get("seconds_cap")
    sampling_rate = int(cfg["sampling_rate"])
    agg = (cfg["classical"].get("aggregation", "mean_proba") or "mean_proba").strip().lower()

    ys: List[int] = []
    ps: List[int] = []
    for item in items:
        X = extract_window_features(item, win, hop, feature_name, seconds_cap, sampling_rate)
        if X.size == 0:
            continue
        if agg == "vote":
            pred_windows = model.predict(X)
            pred = int(np.bincount(pred_windows, minlength=len(CLASS_NAMES)).argmax())
        else:
            scores = _window_scores(model, X)
            pred = int(scores.mean(axis=0).argmax())
        ys.append(int(item["label"]))
        ps.append(pred)
    return ys, ps


def save_artifacts(cfg: dict, model, ys: Sequence[int], ps: Sequence[int], split: str) -> None:
    out_dir = cfg["log"]["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    labels_all = list(range(len(CLASS_NAMES)))
    report = classification_report(
        ys,
        ps,
        labels=labels_all,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    with open(os.path.join(out_dir, f"report_{split}.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    cm = confusion_matrix(ys, ps, labels=labels_all)
    np.savetxt(os.path.join(out_dir, f"confusion_matrix_{split}.csv"), cm, fmt="%d", delimiter=",")

    with open(os.path.join(out_dir, f"predictions_{split}.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["y_true", "y_pred"])
        for y, p in zip(ys, ps):
            writer.writerow([int(y), int(p)])


def train_and_eval(cfg: dict) -> None:
    # train_split_mode / eval_split_mode allow mixing protocols.
    # Example: train on stratified (sees all 3 classes), eval on temporal test
    # (deployment-oriented). This is the fair comparison for classical models
    # that cannot do two-phase training like deep models.
    train_split_mode = cfg.get("train_split_mode") or cfg.get("split_mode", "stratified")
    eval_split_mode = cfg.get("eval_split_mode") or cfg.get("split_mode", "stratified")

    X_train, y_train = build_training_matrix(cfg, split="train", override_split_mode=train_split_mode)

    # For pure temporal training: merge val since SVM has no early-stopping
    # and temporal train-only [0,60]% often lacks minority classes.
    if train_split_mode == "temporal":
        try:
            X_val, y_val = build_training_matrix(cfg, split="val", override_split_mode=train_split_mode)
            X_train = np.concatenate([X_train, X_val], axis=0)
            y_train = np.concatenate([y_train, y_val], axis=0)
        except RuntimeError:
            pass

    n_classes = len(np.unique(y_train))
    if n_classes < 2:
        raise RuntimeError(
            f"Training data has only {n_classes} class(es) under "
            f"train_split_mode='{train_split_mode}'. "
            "Use train_split_mode: stratified to ensure all classes are present."
        )

    model = build_model(cfg)
    model.fit(X_train, y_train)

    eval_split = cfg["classical"].get("eval_split", "test")
    ys, ps = evaluate_filewise(cfg, model, split=eval_split, override_split_mode=eval_split_mode)
    save_artifacts(cfg, model, ys, ps, split=eval_split)

    classes_in_train = sorted(np.unique(y_train).tolist())
    print(f"Train windows: {len(X_train)}  |  classes seen: {[CLASS_NAMES[c] for c in classes_in_train]}")
    print(f"Eval files ({eval_split_mode}/{eval_split}): {len(ys)}")
    print(classification_report(ys, ps, target_names=CLASS_NAMES, digits=4, zero_division=0))

