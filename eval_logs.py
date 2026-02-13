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
    print(classification_report(ys, ps, digits=4))
    print(confusion_matrix(ys, ps))


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
        limit_files=cfg.get("debug", {}).get("limit_files_test"),
        seconds_cap=cfg.get("debug", {}).get("seconds_cap"),
    )
    test_loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=cfg["train"]["num_workers"], pin_memory=True)

    model = ResNet18Small(in_ch=2, num_classes=cfg["num_classes"], temp_feat_dim=6)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)

    print("Test set report (file-wise, mean-agg):")
    evaluate_filewise(test_ds, model, device, batch_size=cfg["train"]["batch_size"], agg="mean")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/logs_stft.yaml")
    ap.add_argument("--ckpt", default="runs/logs_stft/best.pt")
    args = ap.parse_args()
    main(args.config, args.ckpt)
