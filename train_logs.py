import os
import argparse
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from typing import List, Tuple, Optional

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
    # Inverse frequency weights, normalized to mean=1, clipped to a reasonable range
    inv = counts.sum() / counts.astype(np.float32)
    inv = inv / float(inv.mean())
    inv = np.clip(inv, 0.5, 5.0)
    return torch.tensor(inv, dtype=torch.float32)


def evaluate_filewise(
    model,
    ds: LogsTTFDataset,
    device,
    batch_size: int = 32,
    agg: str = "mean",
    max_windows: Optional[int] = None,
    use_amp: bool = False,
):
    """Evaluate by aggregating all windows per file (mean logit or majority-vote)."""
    model.eval()
    ys, ps = [], []
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score
    with torch.no_grad():
        for i in range(len(ds)):
            X, T, y = ds.get_all_windows(i)
            y = int(y.item())
            # batched forward for windows
            # Optionally subsample windows for faster eval
            # Treat max_windows <= 0 as "use all windows"
            if isinstance(max_windows, int) and max_windows > 0 and X.shape[0] > max_windows:
                import numpy as _np
                idx = _np.linspace(0, X.shape[0] - 1, num=max_windows, dtype=int)
                X = X[idx]
                T = T[idx]
            logits_all = []
            n = X.shape[0]
            for s in range(0, n, batch_size):
                xb = X[s:s+batch_size].to(device)
                tb = T[s:s+batch_size].to(device)
                if use_amp:
                    try:
                        from torch.amp import autocast
                        ctx = autocast("cuda")
                    except Exception:
                        from torch.cuda.amp import autocast
                        ctx = autocast()
                    with ctx:
                        lb = model(xb, tb)
                else:
                    lb = model(xb, tb)
                logits_all.append(lb.cpu())
            logits = torch.cat(logits_all, dim=0)  # (W,C)
            if agg == "mean":
                agg_logits = logits.mean(dim=0)
                pred = int(agg_logits.argmax().item())
            else:
                votes = logits.argmax(dim=1).numpy()
                # majority
                pred = int(np.bincount(votes).argmax())
            ys.append(y)
            ps.append(pred)
    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro")
    return acc, f1


def main(cfg_path: str):
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["log"]["out_dir"], exist_ok=True)
    set_seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
            gpu_name = torch.cuda.get_device_name(0)
            print(f"Using GPU: {gpu_name}")
        except Exception:
            pass
    else:
        print("Using CPU")

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
    split_mode = cfg.get("split_mode", "temporal")
    strat = cfg.get("stratified", {"train": 0.7, "val": 0.1, "test": 0.2})
    rnd_seed = cfg.get("random_seed", cfg["train"]["seed"])
    # Optional temporal split ranges from config
    ttf_cfg = cfg.get("temporal_ttf", {}) if split_mode == "temporal" else {}
    ttf_train = tuple(ttf_cfg.get("train", (0.0, 70.0))) if split_mode == "temporal" else (0.0, 0.7)
    ttf_val = tuple(ttf_cfg.get("val", (70.0, 80.0))) if split_mode == "temporal" else (0.7, 0.8)
    ttf_test = tuple(ttf_cfg.get("test", (80.0, 100.1))) if split_mode == "temporal" else (0.8, 1.0)

    use_temp = bool(cfg.get("model", {}).get("use_temp", True))

    train_ds = LogsTTFDataset(
        cfg["data_dir"], cfg["manifest"], split="train",
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        ttf_split=ttf_train,
        split_mode=split_mode,
        train_ratio=strat.get("train", 0.7),
        val_ratio=strat.get("val", 0.1),
        test_ratio=strat.get("test", 0.2),
        min_per_class_val=strat.get("min_per_class_val"),
        min_per_class_test=strat.get("min_per_class_test"),
        random_seed=rnd_seed,
        transform=tr_spec,
        temp_feature_fn=(temp_stats_window if use_temp else None),
        exclude_list=cfg.get("exclude_list"),
        limit_files=cfg.get("debug", {}).get("limit_files_train"),
        seconds_cap=cfg.get("debug", {}).get("seconds_cap"),
    )
    val_ds = LogsTTFDataset(
        cfg["data_dir"], cfg["manifest"], split="val",
        sampling_rate=cfg["sampling_rate"],
        window_seconds=cfg["window_seconds"],
        hop_seconds=cfg["hop_seconds"],
        ttf_split=ttf_val,
        split_mode=split_mode,
        train_ratio=strat.get("train", 0.7),
        val_ratio=strat.get("val", 0.1),
        test_ratio=strat.get("test", 0.2),
        min_per_class_val=strat.get("min_per_class_val"),
        min_per_class_test=strat.get("min_per_class_test"),
        random_seed=rnd_seed,
        transform=te_spec,
        temp_feature_fn=(temp_stats_window if use_temp else None),
        exclude_list=cfg.get("exclude_list"),
        limit_files=cfg.get("debug", {}).get("limit_files_val"),
        seconds_cap=cfg.get("debug", {}).get("seconds_cap"),
    )

    pin_mem = torch.cuda.is_available()
    # DataLoader with persistent workers and prefetch for speed
    dl_common = {
        "batch_size": cfg["train"]["batch_size"],
        "num_workers": cfg["train"]["num_workers"],
        "pin_memory": pin_mem,
    }
    if cfg["train"]["num_workers"] > 0:
        dl_common["persistent_workers"] = True
        dl_common["prefetch_factor"] = int(cfg["train"].get("prefetch_factor", 2))

    # Optional balanced sampling to encourage diagonal learning early
    sampler = None
    if bool(cfg["train"].get("balanced_sampling", False)):
        from collections import Counter
        cnt = Counter([it["label"] for it in train_ds.items])
        # inverse frequency per class
        inv = {k: (sum(cnt.values()) / max(1, v)) for k, v in cnt.items()}
        weights = [inv[it["label"]] for it in train_ds.items]
        weights = torch.tensor(weights, dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        shuffle=(sampler is None),
        sampler=sampler,
        **dl_common,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        **dl_common,
    )

    # Model
    model = ResNet18Small(in_ch=2, num_classes=cfg["num_classes"], temp_feat_dim=(6 if use_temp else 0)).to(device)
    # Optional: initialize from a pretrained checkpoint for fine-tuning
    init_from = cfg["train"].get("init_from")
    if init_from:
        try:
            state = torch.load(init_from, map_location=device, weights_only=True)
            sd = state["model"] if isinstance(state, dict) and "model" in state else state
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"Loaded init weights from {init_from}. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        except Exception as e:
            print(f"Warning: failed to load init_from={init_from}: {e}")

    # Optimizer & scheduler
    use_cw = bool(cfg["train"].get("use_class_weights", True))
    label_smoothing = float(cfg["train"].get("label_smoothing", 0.0))
    if use_cw:
        class_weights = compute_class_weights(cfg["manifest"], LogsTTFDataset.CLASS_MAP).to(device)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    # Optional AMP for faster training on GPU
    use_amp = bool(cfg["train"].get("use_amp", torch.cuda.is_available()))
    scaler = None
    if use_amp and device.type == "cuda":
        try:
            from torch.amp import GradScaler  # PyTorch >= 2.0
            scaler = GradScaler("cuda")
        except Exception:
            from torch.cuda.amp import GradScaler  # fallback for older versions
            scaler = GradScaler()

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
        # Use per-batch stepping for Cosine by setting T_max to total_steps
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    best_metric = -1.0
    best_path = os.path.join(cfg["log"]["out_dir"], "best.pt")
    early_stop = bool(cfg["train"].get("early_stop", True))
    patience = int(cfg["train"].get("early_stop_patience", 0))
    wait = 0

    # for logging curves
    history: List[Tuple[int, float, float, float]] = []  # (epoch, train_loss, val_acc, val_f1)

    eval_every = int(cfg["train"].get("eval_every", 1))
    val_max_windows = cfg["train"].get("val_max_windows")
    if isinstance(val_max_windows, float):
        val_max_windows = int(val_max_windows)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for xb, tb, yb in train_loader:
            xb, tb, yb = xb.to(device), tb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            stepped = False
            if use_amp and scaler is not None:
                try:
                    from torch.amp import autocast  # PyTorch >= 2.0
                    ctx = autocast("cuda")
                except Exception:
                    from torch.cuda.amp import autocast  # fallback
                    ctx = autocast()
                with ctx:
                    logits = model(xb, tb)
                    loss = loss_fn(logits, yb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                stepped = True  # assume stepped; in rare overflow, PyTorch may skip internal step
                scaler.update()
            else:
                logits = model(xb, tb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
                stepped = True
            if scheduler is not None and stepped:
                scheduler.step()
            running += float(loss.item())

        tr_loss = running / max(1, len(train_loader))
        do_eval = ((epoch + 1) % max(1, eval_every) == 0) or (epoch + 1 == cfg["train"]["epochs"]) 
        if do_eval:
            val_acc, val_f1 = evaluate_filewise(
                model, val_ds, device,
                batch_size=cfg["train"]["batch_size"], agg="mean",
                max_windows=val_max_windows,
                use_amp=use_amp and device.type == "cuda",
            )
            metric = val_f1 if cfg["log"]["save_best_by"] == "macro_f1" else val_acc
            print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | train_loss={tr_loss:.4f} | val_acc={val_acc:.4f} | val_f1={val_f1:.4f}")
            history.append((epoch + 1, float(tr_loss), float(val_acc), float(val_f1)))
            if metric > best_metric:
                best_metric = metric
                torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)
                wait = 0
            else:
                wait += 1
                # Early stopping only if enabled and patience > 0
                if early_stop and patience > 0 and wait >= patience:
                    print("Early stopping.")
                    break
        else:
            print(f"Epoch {epoch+1}/{cfg['train']['epochs']} | train_loss={tr_loss:.4f} | val_acc=NA | val_f1=NA")
            history.append((epoch + 1, float(tr_loss), float('nan'), float('nan')))

    print(f"Best model saved to: {best_path}")

    # save history CSV and plot curves if matplotlib exists
    import csv
    hist_csv = os.path.join(cfg["log"]["out_dir"], "train_log.csv")
    with open(hist_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_acc", "val_f1"])
        for row in history:
            writer.writerow(row)

    try:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(6, 4))
        ax1 = fig.add_subplot(111)
        epochs = [e for e, _, _, _ in history]
        tr = [v for _, v, _, _ in history]
        va = [v for _, _, v, _ in history]
        vf = [v for _, _, _, v in history]
        ax1.plot(epochs, tr, label="train_loss")
        ax1.plot(epochs, va, label="val_acc")
        ax1.plot(epochs, vf, label="val_f1")
        # Moving average smoothing for val curves (window=5)
        try:
            import numpy as _np
            def _movavg(x, k=5):
                a = _np.array(x, dtype=float)
                m = _np.isfinite(a).astype(float)
                a = _np.nan_to_num(a, nan=0.0)
                ker = _np.ones(int(max(1, k)), dtype=float)
                num = _np.convolve(a, ker, mode='same')
                den = _np.convolve(m, ker, mode='same')
                den[den == 0] = _np.nan
                return num / den
            va_ma = _movavg(va, k=5)
            vf_ma = _movavg(vf, k=5)
            ax1.plot(epochs, va_ma, label="val_acc_ma", linestyle="--")
            ax1.plot(epochs, vf_ma, label="val_f1_ma", linestyle="--")
        except Exception:
            pass
        ax1.set_xlabel("Epoch")
        ax1.set_title("Training Curves")
        ax1.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(cfg["log"]["out_dir"], "train_curves.png"), dpi=200)
        plt.close(fig)

        # Also save split figures for easier viewing
        try:
            fig_loss = plt.figure(figsize=(6, 3))
            axl = fig_loss.add_subplot(111)
            axl.plot(epochs, tr, label="train_loss", color="#e41a1c")
            axl.set_xlabel("Epoch")
            axl.set_ylabel("Loss")
            axl.set_title("Train Loss")
            axl.grid(True, alpha=0.3)
            fig_loss.tight_layout()
            fig_loss.savefig(os.path.join(cfg["log"]["out_dir"], "train_loss_curve.png"), dpi=200)
            plt.close(fig_loss)

            fig_val = plt.figure(figsize=(6, 3))
            axv = fig_val.add_subplot(111)
            axv.plot(epochs, va, label="val_acc", color="#377eb8")
            axv.plot(epochs, vf, label="val_f1", color="#4daf4a")
            # moving averages if computed
            try:
                axv.plot(epochs, va_ma, label="val_acc_ma", linestyle="--", color="#377eb8", alpha=0.7)
                axv.plot(epochs, vf_ma, label="val_f1_ma", linestyle="--", color="#4daf4a", alpha=0.7)
            except Exception:
                pass
            axv.set_xlabel("Epoch")
            axv.set_ylabel("Score")
            axv.set_ylim(0, 1)
            axv.set_title("Validation Metrics")
            axv.grid(True, alpha=0.3)
            axv.legend()
            fig_val.tight_layout()
            fig_val.savefig(os.path.join(cfg["log"]["out_dir"], "val_metrics_curve.png"), dpi=200)
            plt.close(fig_val)
        except Exception:
            pass
    except Exception:
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/logs_stft.yaml")
    args = ap.parse_args()
    import os as _os
    env_cfg = _os.environ.get("BF_CONFIG")
    cfg_path = env_cfg if env_cfg else args.config
    print(f"Config: {cfg_path}")
    main(cfg_path)
