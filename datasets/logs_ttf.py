import os
import csv
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class LogsTTFDataset(Dataset):
    """
    Windowed dataset from hourly CSV logs with 4 columns per file:
    [vib_x, vib_y, temp_bearing, temp_atm]
    Uses a manifest.csv: file,run_id,ttf_percent,fault_type
    """

    CLASS_MAP = {
        "healthy": 0,
        "degrading": 1,
        "fault": 2,
    }

    def __init__(
        self,
        data_dir: str,
        manifest_path: str,
        split: str,
        sampling_rate: int = 25600,
        window_seconds: float = 1.0,
        hop_seconds: float = 0.5,
        ttf_split: Tuple[float, float] = (0.0, 0.7),
        exclude_list: Optional[str] = None,
        transform=None,
        temp_feature_fn=None,
        limit_files: Optional[int] = None,
        seconds_cap: Optional[float] = None,
    ):
        assert split in {"train", "val", "test"}
        self.data_dir = data_dir
        self.transform = transform
        self.temp_feature_fn = temp_feature_fn
        self.sampling_rate = sampling_rate
        self.win = int(round(window_seconds * sampling_rate))
        self.hop = int(round(hop_seconds * sampling_rate))

        self.seconds_cap = seconds_cap
        self.items = self._build_index(
            manifest_path, split, ttf_split, exclude_list, limit_files
        )

    def _build_index(
        self,
        manifest_path: str,
        split: str,
        ttf_split: Tuple[float, float],
        exclude_list: Optional[str],
        limit_files: Optional[int],
    ) -> List[Dict]:
        # Load exclusions
        exclude = set()
        if exclude_list and os.path.exists(exclude_list):
            with open(exclude_list, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        exclude.add(line)

        # Read manifest
        rows = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            header = f.readline()
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = [p.strip().strip('"') for p in line.split(",")]
                if len(parts) < 4:
                    continue
                r = {
                    "file": parts[0],
                    "run_id": parts[1],
                    "ttf_percent": parts[2],
                    "fault_type": parts[3].lower(),
                }
                if r["file"] in exclude:
                    continue
                rows.append(r)

        # Define split by ttf_percent
        if split == "train":
            lo, hi = 0.0, 70.0
        elif split == "val":
            lo, hi = 70.0, 80.0
        else:
            lo, hi = 80.0, 100.1  # include 100

        lo_ttf, hi_ttf = lo, hi
        items: List[Dict] = []
        for r in rows:
            p = float(r["ttf_percent"]) if r["ttf_percent"] != "" else 0.0
            if not (lo_ttf <= p < hi_ttf):
                continue
            path = os.path.join(self.data_dir, r["file"])
            label_name = r["fault_type"].strip().lower()
            if label_name not in self.CLASS_MAP:
                # skip unknown labels in this baseline
                continue
            label = self.CLASS_MAP[label_name]
            # record-level entry; windows will be generated on __getitem__ demand
            items.append({"path": path, "label": label})
        if limit_files is not None:
            items = items[: int(limit_files)]
        return items

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    def _read_csv_fast(path: str, max_rows: Optional[int] = None) -> np.ndarray:
        # Memory-map read for speed; expect 4 columns, comma-separated, no header
        # Fallback to numpy loadtxt for simplicity
        kwargs = {"delimiter": ","}
        if max_rows is not None:
            kwargs["max_rows"] = int(max_rows)
        return np.loadtxt(path, **kwargs)

    def _make_windows(self, n: int) -> List[Tuple[int, int]]:
        idx = []
        if n < self.win:
            return []
        pos = 0
        while pos + self.win <= n:
            idx.append((pos, pos + self.win))
            pos += self.hop
        return idx

    def __getitem__(self, i: int):
        item = self.items[i]
        cap = int(self.seconds_cap * self.sampling_rate) if self.seconds_cap else None
        arr = self._read_csv_fast(item["path"], max_rows=cap)  # (N,4)
        # split channels
        vib = arr[:, :2].astype(np.float32)  # (N,2)
        temp = arr[:, 2:].astype(np.float32)  # (N,2)

        windows = self._make_windows(vib.shape[0])
        if not windows:
            raise IndexError(f"File too short for window: {item['path']}")

        # Select a random window for training, first window for val/test
        if self.transform and hasattr(self.transform, "training") and self.transform.training:
            widx = np.random.randint(0, len(windows))
        else:
            widx = 0
        s, e = windows[widx]
        vib_w = vib[s:e]  # (win,2)
        temp_w = temp[s:e]  # (win,2)

        # Apply transform for vibration (e.g., STFT -> 2xFxT)
        x = self.transform(vib_w) if self.transform else vib_w

        # Temperature features per window
        tfeat = self.temp_feature_fn(temp_w) if self.temp_feature_fn else np.zeros(6, dtype=np.float32)

        y = item["label"]
        return x, torch.tensor(tfeat, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    # --- Helpers for multi-window evaluation/inference ---
    def get_all_windows(self, i: int):
        item = self.items[i]
        cap = int(self.seconds_cap * self.sampling_rate) if self.seconds_cap else None
        arr = self._read_csv_fast(item["path"], max_rows=cap)  # (N,4)
        vib = arr[:, :2].astype(np.float32)
        temp = arr[:, 2:].astype(np.float32)
        windows = self._make_windows(vib.shape[0])
        X = []
        T = []
        for s, e in windows:
            vib_w = vib[s:e]
            temp_w = temp[s:e]
            x = self.transform(vib_w) if self.transform else vib_w
            t = self.temp_feature_fn(temp_w) if self.temp_feature_fn else np.zeros(6, dtype=np.float32)
            X.append(x)
            T.append(t)
        if not X:
            raise IndexError(f"No windows for file: {item['path']}")
        X = torch.stack(X, dim=0)  # (W,2,H,W)
        T = torch.tensor(np.stack(T, axis=0), dtype=torch.float32)  # (W,6)
        y = torch.tensor(item["label"], dtype=torch.long)
        return X, T, y
