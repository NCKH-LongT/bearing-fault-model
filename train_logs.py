import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from datasets.logs_ttf import LogsTTFDataset
from features.spectrogram import SpectrogramTransform
from features.temp_features import temp_stats_window
from models.resnet2d import ResNet18Small


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_class_weights(manifest_path: str, mapping: dict) -> torch.Tensor:
    import csv
    counts = np.zeros(len(mapping), dtype=np.int64)
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            name = r["fault_type"].strip().lower()
            if name in mapping:
                counts[mapping[name]] += 1
    counts = np.maximum(counts, 1)
    weights = counts.sum() / counts.astype(np.float32)
    return torch.tensor(weights, dtype=torch.float32)


def evaluate(model, loader, device):
    model.eval()
    all_y, all_p = [], []
    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    with torch.no_grad():
        for xb, tb, yb in loader:
            xb = xb.to(device)
            tb = tb.to(device)
            yb = yb.to(device)
            logits = model(xb, tb)
            loss = loss_fn(logits, yb)
            total_loss += float(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            all_p.append(preds)
            all_y.append(yb.cpu().numpy())
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    y = np.concatenate(all_y)
    p = np.concatenate(all_p)
    acc = accuracy_score(y, p)
    f1 = f1_score(y, p, average="macro")
    return total_loss / len(loader.dataset), acc, f1


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["log"]["out_dir"], exist_ok=True)
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    tr_spec = SpectrogramTransform(
        n_fft=cfg["stft"]["n_fft"],
        hop_length=cfg["stft"]["hop_length"],
        window=cfg["stft"]["window"],
        log_add=cfg["stft"]["log_add"],
        target_size=tuple(cfg["input_size"]),
        training=True,
    )
    te_spec = SpectrogramTransform(
        n_fft=cfg["stft"]["n_fft"],
        hop_length=cfg["stft"]["hop_length"],
        window=cfg["stft"]["window"],
        log_add=cfg["stft"]["log_add"],
        target_size=tuple(cfg["input_size"]),
        training=False,
    )

    # Datasets
    train_ds = LogsTTFDataset(
        cfg["data_dir"], cfg["manifest"], split="train",
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        transform=tr_spec,
        temp_feature_fn=temp_stats_window,
        exclude_list=cfg.get("exclude_list"),
        limit_files=cfg.get("debug", {}).get("limit_files_train"),
        seconds_cap=cfg.get("debug", {}).get("seconds_cap"),
    )
    val_ds = LogsTTFDataset(
        cfg["data_dir"], cfg["manifest"], split="val",
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        transform=te_spec,
        temp_feature_fn=temp_stats_window,
        exclude_list=cfg.get("exclude_list"),
        limit_files=cfg.get("debug", {}).get("limit_files_val"),
        seconds_cap=cfg.get("debug", {}).get("seconds_cap"),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, num_workers=cfg["train"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

    # Model
    model = ResNet18Small(in_ch=2, num_classes=cfg["num_classes"], temp_feat_dim=6).to(device)

    # Optimizer & scheduler
    class_weights = compute_class_weights(cfg["manifest"], LogsTTFDataset.CLASS_MAP).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    total_steps = cfg["train"]["epochs"] * max(1, len(train_loader))
    if cfg["optim"]["use_onecycle"] and total_steps > 0:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=cfg["train"]["lr"],
            steps_per_epoch=max(1, len(train_loader)),
            epochs=cfg["train"]["epochs"],
            pct_start=cfg["optim"].get("pct_start", 0.3),
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg["train"]["epochs"]))

    best_metric = -1.0
    best_path = os.path.join(cfg["log"]["out_dir"], "best.pt")
    patience = cfg["train"]["early_stop_patience"]
    wait = 0

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for xb, tb, yb in train_loader:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb, tb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            running += float(loss.item())

        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        metric = val_f1 if cfg["log"]["save_best_by"] == "macro_f1" else val_acc
        print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | train_loss={running/max(1,len(train_loader)):.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")
        if metric > best_metric:
            best_metric = metric
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping.")
                break

    print(f"Best model saved to: {best_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/logs_stft.yaml")
    args = ap.parse_args()
    main(args.config)
