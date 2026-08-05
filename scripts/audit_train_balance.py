#!/usr/bin/env python3
"""Audit balanced sampling and per-class gradient flow using train files only."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets.logs_ttf import LogsTTFDataset
from features.spectrogram import SpectrogramTransform
from features.temp_features import StandardizedTempFeature, resolve_temp_feature
from models.resnet2d import ResNet18Small


CLASS_NAMES = ("healthy", "degrading", "fault")


def gradient_stats(parameters) -> dict[str, float | int]:
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    squared = sum(float(torch.sum(gradient ** 2)) for gradient in gradients)
    count = sum(gradient.numel() for gradient in gradients)
    return {
        "parameters_with_gradient": count,
        "l2": squared ** 0.5,
        "rms": (squared / count) ** 0.5 if count else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Run-local config containing train-fitted scaler stats.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--files-per-class", type=int, default=8)
    parser.add_argument("--sampler-draws", type=int, default=10000)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    temp_cfg = cfg["model"]["temp_feature"]
    temp_fn, temp_dim = resolve_temp_feature(temp_cfg.get("type", "stats6"))
    norm_cfg = temp_cfg.get("normalization", {})
    if norm_cfg.get("mean") is None or norm_cfg.get("std") is None:
        raise SystemExit("Run-local config is missing train-fitted normalization stats")
    temp_fn = StandardizedTempFeature(temp_fn, norm_cfg["mean"], norm_cfg["std"])
    transform = SpectrogramTransform(
        n_fft=cfg["stft"]["n_fft"],
        hop_length=cfg["stft"]["hop_length"],
        window=cfg["stft"]["window"],
        log_add=cfg["stft"]["log_add"],
        target_size=tuple(cfg["input_size"]),
        training=False,
    )
    strat = cfg["stratified"]
    dataset = LogsTTFDataset(
        cfg["data_dir"],
        cfg["manifest"],
        split="train",
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        split_mode=cfg["split_mode"],
        train_ratio=strat["train"],
        val_ratio=strat["val"],
        test_ratio=strat["test"],
        min_per_class_val=strat.get("min_per_class_val"),
        min_per_class_test=strat.get("min_per_class_test"),
        random_seed=cfg["random_seed"],
        transform=transform,
        temp_feature_fn=temp_fn,
        temp_feat_dim=temp_dim,
        cache_dir=cfg.get("cache_dir"),
    )

    file_counts = Counter(int(item["label"]) for item in dataset.items)
    weights = np.asarray([1.0 / file_counts[int(item["label"])] for item in dataset.items], dtype=float)
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(20260805)
    draws = rng.choice(len(dataset.items), size=args.sampler_draws, replace=True, p=probabilities)
    sampled_counts = Counter(int(dataset.items[index]["label"]) for index in draws)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet18Small(in_ch=2, num_classes=cfg["num_classes"], temp_feat_dim=temp_dim, use_vibration=True).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state["model"] if isinstance(state, dict) else state)
    model.train()
    groups = {
        "vibration_backbone": list(model.stem.parameters()) + list(model.layer1.parameters()) + list(model.layer2.parameters()) + list(model.layer3.parameters()) + list(model.layer4.parameters()),
        "temperature_branch": list(model.temp_proj.parameters()),
        "classifier": list(model.classifier.parameters()),
    }
    gradients = {}
    for label, name in enumerate(CLASS_NAMES):
        indices = [index for index, item in enumerate(dataset.items) if int(item["label"]) == label]
        if len(indices) > args.files_per_class:
            positions = np.linspace(0, len(indices) - 1, args.files_per_class, dtype=int)
            indices = [indices[position] for position in positions]
        samples = [dataset[index] for index in indices]
        vibration = torch.stack([sample[0] for sample in samples]).to(device)
        temperature = torch.stack([sample[1] for sample in samples]).to(device)
        target = torch.stack([sample[2] for sample in samples]).to(device)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(vibration, temperature), target)
        loss.backward()
        gradients[name] = {
            "files": len(indices),
            "loss": float(loss.detach()),
            "gradient": {group_name: gradient_stats(parameters) for group_name, parameters in groups.items()},
        }

    report = {
        "protocol": "train split only; locked test not instantiated",
        "train_file_counts": {CLASS_NAMES[label]: file_counts[label] for label in range(3)},
        "balanced_sampler_empirical_counts": {CLASS_NAMES[label]: sampled_counts[label] for label in range(3)},
        "sampler_draws": args.sampler_draws,
        "per_class_gradient_audit": gradients,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Train sampler and gradient audit",
        "",
        "Train split only; the locked test split was not instantiated.",
        "",
        "| Class | Train files | Empirical balanced draws | Draw share |",
        "|---|---:|---:|---:|",
    ]
    for label, name in enumerate(CLASS_NAMES):
        lines.append(f"| {name} | {file_counts[label]} | {sampled_counts[label]} | {sampled_counts[label] / args.sampler_draws:.4f} |")
    lines += ["", "| Class | Files | Loss | Vibration grad RMS | Temperature grad RMS | Classifier grad RMS |", "|---|---:|---:|---:|---:|---:|"]
    for name in CLASS_NAMES:
        row = gradients[name]
        grad = row["gradient"]
        lines.append(
            f"| {name} | {row['files']} | {row['loss']:.4f} | {grad['vibration_backbone']['rms']:.6f} | "
            f"{grad['temperature_branch']['rms']:.6f} | {grad['classifier']['rms']:.6f} |"
        )
    (args.output_dir / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output_dir / "audit.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
