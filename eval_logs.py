import os
import argparse
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.logs_ttf import LogsTTFDataset
from features.spectrogram import SpectrogramTransform
from features.temp_features import temp_stats_window
from models.resnet2d import ResNet18Small


def evaluate_subset(loader, model, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, tb, yb in loader:
            xb, tb = xb.to(device), tb.to(device)
            logits = model(xb, tb)
            pred = logits.argmax(1).cpu().numpy()
            ys.append(yb.numpy())
            ps.append(pred)
    from sklearn.metrics import classification_report, confusion_matrix
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    print(classification_report(y, p, digits=4))
    print(confusion_matrix(y, p))


def main(cfg_path: str, ckpt_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    te_spec = SpectrogramTransform(
        n_fft=cfg["stft"]["n_fft"],
        hop_length=cfg["stft"]["hop_length"],
        window=cfg["stft"]["window"],
        log_add=cfg["stft"]["log_add"],
        target_size=tuple(cfg["input_size"]),
        training=False,
    )

    test_ds = LogsTTFDataset(
        cfg["data_dir"], cfg["manifest"], split="test",
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        transform=te_spec,
        temp_feature_fn=temp_stats_window,
        exclude_list=cfg.get("exclude_list"),
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

    model = ResNet18Small(in_ch=2, num_classes=cfg["num_classes"], temp_feat_dim=6)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)

    print("Test set report:")
    evaluate_subset(test_loader, model, device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/logs_stft.yaml")
    ap.add_argument("--ckpt", default="runs/logs_stft/best.pt")
    args = ap.parse_args()
    main(args.config, args.ckpt)

