#!/usr/bin/env python3
"""Five-seed validation robustness and efficiency benchmark for normalized fusion."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from classical_baselines.pipeline import make_windows, read_signal_csv
from datasets.logs_ttf import LogsTTFDataset
from features.spectrogram import SpectrogramTransform
from features.temp_features import StandardizedTempFeature, resolve_temp_feature
from models.resnet2d import ResNet18Small


CLASS_NAMES = ("healthy", "degrading", "fault")
SEEDS = (42, 43, 44, 45, 46)
CONDITIONS = ("clean", "noise_20db", "noise_10db", "missing_vib_x", "missing_vib_y", "missing_temperature", "temperature_drift_plus_2c")


def mean_std(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
    }


def perturb(window: np.ndarray, condition: str, rng: np.random.Generator) -> np.ndarray:
    output = np.asarray(window, dtype=np.float32).copy()
    if condition.startswith("noise_"):
        snr_db = float(condition.split("_")[1].removesuffix("db"))
        for channel in (0, 1):
            power = float(np.mean(output[:, channel] ** 2))
            noise_std = np.sqrt(power / (10.0 ** (snr_db / 10.0)) + 1e-12)
            output[:, channel] += rng.normal(0.0, noise_std, len(output)).astype(np.float32)
    elif condition == "missing_vib_x":
        output[:, 0] = 0.0
    elif condition == "missing_vib_y":
        output[:, 1] = 0.0
    elif condition == "temperature_drift_plus_2c":
        output[:, 2:4] += 2.0
    return output


def build_validation_dataset(cfg: dict) -> LogsTTFDataset:
    strat = cfg["stratified"]
    return LogsTTFDataset(
        cfg["data_dir"], cfg["manifest"], split="val",
        sampling_rate=cfg["sampling_rate"], window_seconds=cfg["window_seconds"], hop_seconds=cfg["hop_seconds"],
        split_mode=cfg["split_mode"], train_ratio=strat["train"], val_ratio=strat["val"], test_ratio=strat["test"],
        min_per_class_val=strat.get("min_per_class_val"), min_per_class_test=strat.get("min_per_class_test"),
        random_seed=cfg["random_seed"], cache_dir=cfg.get("cache_dir"), transform=None,
        temp_feature_fn=None, temp_feat_dim=0,
    )


def prepare_condition(cfg: dict, dataset: LogsTTFDataset, condition: str, maximum: int) -> tuple[list[tuple[torch.Tensor, torch.Tensor, int]], float, int]:
    spec = SpectrogramTransform(
        n_fft=cfg["stft"]["n_fft"], hop_length=cfg["stft"]["hop_length"], window=cfg["stft"]["window"],
        log_add=cfg["stft"]["log_add"], target_size=tuple(cfg["input_size"]), training=False,
    )
    temp_cfg = cfg["model"]["temp_feature"]
    raw_temp_fn, _ = resolve_temp_feature(temp_cfg.get("type", "stats6"))
    norm_cfg = temp_cfg["normalization"]
    temp_fn = StandardizedTempFeature(raw_temp_fn, norm_cfg["mean"], norm_cfg["std"])
    win = int(round(float(cfg["window_seconds"]) * int(cfg["sampling_rate"])))
    hop = int(round(float(cfg["hop_seconds"]) * int(cfg["sampling_rate"])))
    prepared = []
    preprocessing_seconds = 0.0
    window_count = 0
    for file_index, item in enumerate(dataset.items):
        signal = read_signal_csv(item["path"], cache_dir=cfg.get("cache_dir"))
        windows = make_windows(len(signal), win, hop)
        if len(windows) > maximum:
            indices = np.linspace(0, len(windows) - 1, maximum, dtype=int)
            windows = [windows[index] for index in indices]
        vibration, temperature = [], []
        for window_index, (start, end) in enumerate(windows):
            rng = np.random.default_rng(20260805 + file_index * 1000 + window_index)
            window = perturb(signal[start:end], condition, rng)
            started = time.perf_counter()
            vibration.append(spec(window[:, :2]))
            temp_vector = temp_fn(window[:, 2:4])
            if condition == "missing_temperature":
                temp_vector = np.zeros_like(temp_vector)
            temperature.append(temp_vector)
            preprocessing_seconds += time.perf_counter() - started
        prepared.append((torch.stack(vibration), torch.tensor(np.stack(temperature), dtype=torch.float32), int(item["label"])))
        window_count += len(windows)
    return prepared, preprocessing_seconds, window_count


def load_models(root: Path, device: torch.device) -> list[tuple[int, ResNet18Small]]:
    models = []
    for seed in SEEDS:
        run_dir = root / f"seed_{seed}"
        cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
        temp_fn, temp_dim = resolve_temp_feature(cfg["model"]["temp_feature"].get("type", "stats6"))
        del temp_fn
        model = ResNet18Small(in_ch=2, num_classes=cfg["num_classes"], temp_feat_dim=temp_dim, use_vibration=True)
        state = torch.load(run_dir / "best.pt", map_location=device, weights_only=True)
        model.load_state_dict(state["model"] if isinstance(state, dict) else state)
        model.to(device).eval()
        if device.type == "cuda":
            model.to(memory_format=torch.channels_last)
        models.append((seed, model))
    return models


def evaluate(model: ResNet18Small, prepared, device: torch.device, batch_size: int) -> dict:
    truth, predicted = [], []
    with torch.inference_mode():
        for vibration, temperature, label in prepared:
            outputs = []
            for start in range(0, len(vibration), batch_size):
                xb = vibration[start : start + batch_size].to(device)
                tb = temperature[start : start + batch_size].to(device)
                if device.type == "cuda":
                    xb = xb.contiguous(memory_format=torch.channels_last)
                    with torch.amp.autocast("cuda"):
                        outputs.append(model(xb, tb).float().cpu())
                else:
                    outputs.append(model(xb, tb).cpu())
            prediction = int(torch.cat(outputs).mean(dim=0).argmax())
            truth.append(label)
            predicted.append(prediction)
    truth_array, predicted_array = np.asarray(truth), np.asarray(predicted)
    per_class = f1_score(truth_array, predicted_array, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(truth_array, predicted_array)),
        "macro_f1": float(f1_score(truth_array, predicted_array, labels=[0, 1, 2], average="macro", zero_division=0)),
        "class_f1": {CLASS_NAMES[index]: float(per_class[index]) for index in range(3)},
    }


def efficiency(model, prepared, checkpoint: Path, device: torch.device, repeats: int) -> dict:
    vibration, temperature, _ = prepared[0]
    xb = vibration.to(device)
    tb = temperature.to(device)
    if device.type == "cuda":
        xb = xb.contiguous(memory_format=torch.channels_last)
    for _ in range(10):
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    model(xb, tb)
            else:
                model(xb, tb)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        with torch.inference_mode():
            if device.type == "cuda":
                with torch.amp.autocast("cuda"):
                    model(xb, tb)
                torch.cuda.synchronize()
            else:
                model(xb, tb)
        timings.append(time.perf_counter() - started)
    return {
        "device": str(device),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "batch_windows": len(xb),
        "model_ms_per_window": 1000.0 * float(np.mean(timings)) / len(xb),
        "model_windows_per_second": len(xb) / float(np.mean(timings)),
        "peak_device_memory_bytes": int(torch.cuda.max_memory_allocated()) if device.type == "cuda" else None,
        "latency_repeats": repeats,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("runs/revision_multimodal_normalized_validation"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper/revision_artifacts/deep_robustness_efficiency"))
    parser.add_argument("--max-windows-per-file", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--latency-repeats", type=int, default=100)
    args = parser.parse_args()

    run_root = ROOT / args.run_root
    cfg = yaml.safe_load((run_root / "seed_42/config.yaml").read_text(encoding="utf-8"))
    dataset = build_validation_dataset(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_models(run_root, device)
    condition_runs = {}
    clean_preprocessing = None
    clean_prepared = None
    for condition in CONDITIONS:
        prepared, preprocessing_seconds, window_count = prepare_condition(cfg, dataset, condition, args.max_windows_per_file)
        if condition == "clean":
            clean_preprocessing = {
                "cpu_preprocessing_ms_per_window": 1000.0 * preprocessing_seconds / window_count,
                "cpu_preprocessing_windows_per_second": window_count / preprocessing_seconds,
            }
            clean_prepared = prepared
        condition_runs[condition] = [
            {"seed": seed, **evaluate(model, prepared, device, args.batch_size)} for seed, model in models
        ]
        del prepared

    aggregate = {}
    clean_macro = mean_std([row["macro_f1"] for row in condition_runs["clean"]])["mean"]
    for condition, runs in condition_runs.items():
        aggregate[condition] = {
            "accuracy": mean_std([row["accuracy"] for row in runs]),
            "macro_f1": mean_std([row["macro_f1"] for row in runs]),
            "class_f1": {
                name: mean_std([row["class_f1"][name] for row in runs]) for name in CLASS_NAMES
            },
        }
        aggregate[condition]["macro_f1_drop_from_clean"] = clean_macro - aggregate[condition]["macro_f1"]["mean"]

    assert clean_prepared is not None and clean_preprocessing is not None
    model_efficiency = efficiency(models[0][1], clean_prepared, run_root / "seed_42/best.pt", device, args.latency_repeats)
    efficiency_report = {**clean_preprocessing, **model_efficiency}
    efficiency_report["estimated_end_to_end_ms_per_32_window_file"] = 32.0 * (
        efficiency_report["cpu_preprocessing_ms_per_window"] + efficiency_report["model_ms_per_window"]
    )
    efficiency_report["estimated_files_per_second_32_windows"] = 1000.0 / efficiency_report["estimated_end_to_end_ms_per_32_window_file"]
    report = {
        "protocol": "five-seed validation-only robustness; locked test not instantiated",
        "conditions": aggregate,
        "per_seed": condition_runs,
        "efficiency_seed": 42,
        "efficiency": efficiency_report,
    }
    output_dir = ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Deep multimodal robustness and efficiency",
        "",
        "Five-seed validation only; raw-signal perturbations were applied before STFT and the locked test split was not instantiated.",
        "",
        "| Condition | Accuracy mean ± std | Macro-F1 mean ± std | Drop | H F1 | D F1 | F F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        row = aggregate[condition]
        lines.append(
            f"| {condition} | {row['accuracy']['mean']:.4f} ± {row['accuracy']['std']:.4f} | "
            f"{row['macro_f1']['mean']:.4f} ± {row['macro_f1']['std']:.4f} | {row['macro_f1_drop_from_clean']:.4f} | "
            f"{row['class_f1']['healthy']['mean']:.4f} | {row['class_f1']['degrading']['mean']:.4f} | {row['class_f1']['fault']['mean']:.4f} |"
        )
    lines += [
        "",
        "## Efficiency (seed 42)",
        "",
        f"- CPU STFT + temperature preprocessing: `{efficiency_report['cpu_preprocessing_ms_per_window']:.3f} ms/window`.",
        f"- {device} model forward: `{efficiency_report['model_ms_per_window']:.3f} ms/window` (`{efficiency_report['model_windows_per_second']:.1f}` windows/s), batch {efficiency_report['batch_windows']}.",
        f"- Estimated 32-window end-to-end: `{efficiency_report['estimated_end_to_end_ms_per_32_window_file']:.2f} ms/file` (`{efficiency_report['estimated_files_per_second_32_windows']:.2f}` files/s), excluding I/O.",
        f"- Parameters: `{efficiency_report['parameters']}`; checkpoint: `{efficiency_report['checkpoint_bytes'] / 1024**2:.2f} MiB`.",
        f"- Peak CUDA allocated memory: `{efficiency_report['peak_device_memory_bytes'] / 1024**2:.2f} MiB`." if efficiency_report["peak_device_memory_bytes"] is not None else "- Peak device memory: unavailable on CPU.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_dir / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
