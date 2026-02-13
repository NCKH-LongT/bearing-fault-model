import os
import argparse
import yaml
import torch
import numpy as np

from datasets.logs_ttf import LogsTTFDataset
from features.spectrogram import SpectrogramTransform
from features.temp_features import temp_stats_window
from models.resnet2d import ResNet18Small


def load_model(cfg, ckpt_path, device):
    model = ResNet18Small(in_ch=2, num_classes=cfg["num_classes"], temp_feat_dim=6)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


def slide_predict_file(path, cfg, model, device, batch_size=32):
    # Build a temporary dataset-like helper to reuse windowing/feature code
    helper = LogsTTFDataset(
        data_dir=os.path.dirname(path) or ".",
        manifest_path=cfg["manifest"],  # not used for file list in this helper call
        split="test",
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        transform=SpectrogramTransform(
            n_fft=cfg["stft"]["n_fft"],
            hop_length=cfg["stft"]["hop_length"],
            window=cfg["stft"]["window"],
            log_add=cfg["stft"]["log_add"],
            target_size=tuple(cfg["input_size"]),
            training=False,
        ),
        temp_feature_fn=temp_stats_window,
        limit_files=0,
    )
    # Monkey-patch one item for this path
    helper.items = [{"path": path, "label": 0}]
    X, T, _ = helper.get_all_windows(0)
    outs = []
    with torch.no_grad():
        for s in range(0, X.shape[0], batch_size):
            xb = X[s:s+batch_size].to(device)
            tb = T[s:s+batch_size].to(device)
            lb = model(xb, tb)
            outs.append(lb.softmax(dim=1).cpu())
    probs = torch.cat(outs, dim=0)  # (W,C)
    mean_prob = probs.mean(dim=0)   # (C,)
    pred = int(mean_prob.argmax().item())
    conf = float(mean_prob.max().item())
    return pred, conf, mean_prob.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/logs_stft.yaml")
    ap.add_argument("--ckpt", default="runs/logs_stft/best.pt")
    ap.add_argument("inputs", nargs="+", help="Path(s) to CSV file(s)")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, args.ckpt, device)

    inv_map = {v: k for k, v in LogsTTFDataset.CLASS_MAP.items()}
    for p in args.inputs:
        pred, conf, dist = slide_predict_file(p, cfg, model, device)
        label = inv_map.get(pred, str(pred))
        print(f"{os.path.basename(p)} -> {label} (conf={conf:.3f})")


if __name__ == "__main__":
    main()

