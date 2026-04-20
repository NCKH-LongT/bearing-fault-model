import os
import csv
from typing import Tuple, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from functools import lru_cache


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
        split_mode: str = "temporal",  # "temporal" | "stratified"
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        test_ratio: float = 0.2,
        random_seed: int = 42,
        min_per_class_val: Optional[int] = None,
        min_per_class_test: Optional[int] = None,
        exclude_list: Optional[str] = None,
        transform=None,
        temp_feature_fn=None,
        temp_feat_dim: Optional[int] = None,
        temp_context_seconds: Optional[float] = None,
        temp_context_causal: bool = True,
        limit_files: Optional[int] = None,
        seconds_cap: Optional[float] = None,
    ):
        assert split in {"train", "val", "test"}
        self.data_dir = data_dir
        self.transform = transform
        self.temp_feature_fn = temp_feature_fn
        if temp_feat_dim is None:
            self.temp_feat_dim = 6 if self.temp_feature_fn else 0
        else:
            self.temp_feat_dim = int(max(0, temp_feat_dim))
        self.sampling_rate = sampling_rate
        self.win = int(round(window_seconds * sampling_rate))
        self.hop = int(round(hop_seconds * sampling_rate))

        self.seconds_cap = seconds_cap
        self.temp_context_causal = bool(temp_context_causal)
        if temp_context_seconds is None:
            self.temp_context = 0
        else:
            self.temp_context = int(round(float(temp_context_seconds) * sampling_rate))
        self.items = self._build_index(
            manifest_path, split, ttf_split, split_mode,
            train_ratio, val_ratio, test_ratio, random_seed,
            min_per_class_val, min_per_class_test,
            exclude_list, limit_files
        )

    def _slice_temp_context(self, temp: np.ndarray, s: int, e: int) -> np.ndarray:
        """
        Return a temperature segment to compute temp features for a window [s,e).

        If temp_context is 0 (disabled), returns temp[s:e].
        If enabled:
          - causal=True: use only history up to e, i.e. [max(0, e-ctx), e)
          - causal=False: use centered context around the window midpoint
        """
        n = int(temp.shape[0])
        if self.temp_context <= 0 or self.temp_context <= (e - s):
            return temp[s:e]

        ctx = int(self.temp_context)
        if self.temp_context_causal:
            a = max(0, int(e) - ctx)
            b = int(e)
            return temp[a:b]

        mid = int((s + e) // 2)
        a = max(0, mid - ctx // 2)
        b = min(n, a + ctx)
        # If we hit the end, shift start left to keep length ~ctx
        a = max(0, b - ctx)
        return temp[a:b]

    def _build_index(
        self,
        manifest_path: str,
        split: str,
        ttf_split: Tuple[float, float],
        split_mode: str,
        train_ratio: float,
        val_ratio: float,
        test_ratio: float,
        random_seed: int,
        min_per_class_val: Optional[int],
        min_per_class_test: Optional[int],
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

        items: List[Dict] = []
        if split_mode == "temporal":
            # Define split by ttf_percent
            # Prefer provided ttf_split from caller (config), fallback to defaults
            if ttf_split is not None and len(ttf_split) == 2:
                lo, hi = float(ttf_split[0]), float(ttf_split[1])
            else:
                if split == "train":
                    lo, hi = 0.0, 70.0
                elif split == "val":
                    lo, hi = 70.0, 80.0
                else:
                    lo, hi = 80.0, 100.1  # include 100

            lo_ttf, hi_ttf = lo, hi
            for r in rows:
                p = float(r["ttf_percent"]) if r["ttf_percent"] != "" else 0.0
                if not (lo_ttf <= p < hi_ttf):
                    continue
                path = os.path.join(self.data_dir, r["file"])
                label_name = r["fault_type"].strip().lower()
                if label_name not in self.CLASS_MAP:
                    continue
                label = self.CLASS_MAP[label_name]
                try:
                    ttf = float(r["ttf_percent"]) if r["ttf_percent"] != "" else 0.0
                except Exception:
                    ttf = 0.0
                items.append({
                    "path": path,
                    "label": label,
                    "file": r["file"],
                    "ttf_percent": ttf,
                })
        else:
            # Stratified split by class across the whole manifest
            assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6, "Ratios must sum to 1.0"
            by_cls: Dict[str, List[Dict]] = {}
            for r in rows:
                name = r["fault_type"].strip().lower()
                if name not in self.CLASS_MAP:
                    continue
                by_cls.setdefault(name, []).append(r)
            rng = np.random.RandomState(random_seed)
            selected: List[Dict] = []
            for name, lst in by_cls.items():
                idx = np.arange(len(lst))
                rng.shuffle(idx)
                n = len(idx)
                n_tr = int(round(n * train_ratio))
                n_va = int(round(n * val_ratio))
                # ensure coverage
                mpv = int(min_per_class_val) if (min_per_class_val is not None) else 0
                mpt = int(min_per_class_test) if (min_per_class_test is not None) else 0
                if n_va < mpv:
                    # borrow from train first, then test
                    extra = mpv - n_va
                    take_tr = min(extra, n_tr)
                    n_tr -= take_tr
                    extra -= take_tr
                    n_va += take_tr
                    if extra > 0:
                        # increase total val by reducing test later
                        n_va += extra
                n_te = max(0, n - n_tr - n_va)
                if n_te < mpt:
                    # borrow from train first, then val
                    extra = mpt - n_te
                    take_tr = min(extra, n_tr)
                    n_tr -= take_tr
                    extra -= take_tr
                    n_te += take_tr
                    if extra > 0 and n_va > extra:
                        n_va -= extra
                        n_te += extra
                # final clamp to valid bounds
                n_tr = max(0, min(n_tr, n))
                n_va = max(0, min(n_va, n - n_tr))
                n_te = max(0, n - n_tr - n_va)
                if split == "train":
                    use = idx[:n_tr]
                elif split == "val":
                    use = idx[n_tr:n_tr+n_va]
                else:
                    use = idx[n_tr+n_va:]
                for k in use:
                    rr = lst[int(k)]
                    try:
                        ttf = float(rr["ttf_percent"]) if rr["ttf_percent"] != "" else 0.0
                    except Exception:
                        ttf = 0.0
                    selected.append({
                        "path": os.path.join(self.data_dir, rr["file"]),
                        "label": self.CLASS_MAP[name],
                        "file": rr["file"],
                        "ttf_percent": ttf,
                    })
            items = selected
        if limit_files is not None:
            items = items[: int(limit_files)]
        return items

    def __len__(self) -> int:
        return len(self.items)

    @staticmethod
    @lru_cache(maxsize=128)
    def _read_csv_cached(path: str, max_rows_key: Optional[int]) -> np.ndarray:
        # Cached CSV reader to avoid re-reading the same file multiple times across epochs
        kwargs = {"delimiter": ","}
        if max_rows_key is not None and max_rows_key >= 0:
            kwargs["max_rows"] = int(max_rows_key)
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
        cap_key = int(cap) if cap is not None else -1
        arr = self._read_csv_cached(item["path"], cap_key)  # (N,4)
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
        temp_w = self._slice_temp_context(temp, s, e)  # (ctx,2) or (win,2)

        # Apply transform for vibration (e.g., STFT -> 2xFxT)
        x = self.transform(vib_w) if self.transform else vib_w

        # Temperature features per window
        if self.temp_feature_fn:
            tfeat = self.temp_feature_fn(temp_w)
        else:
            tfeat = np.zeros(self.temp_feat_dim, dtype=np.float32)

        y = item["label"]
        return x, torch.tensor(tfeat, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    # --- Helpers for multi-window evaluation/inference ---
    def get_all_windows(self, i: int):
        item = self.items[i]
        cap = int(self.seconds_cap * self.sampling_rate) if self.seconds_cap else None
        cap_key = int(cap) if cap is not None else -1
        arr = self._read_csv_cached(item["path"], cap_key)  # (N,4)
        vib = arr[:, :2].astype(np.float32)
        temp = arr[:, 2:].astype(np.float32)
        windows = self._make_windows(vib.shape[0])
        X = []
        T = []
        for s, e in windows:
            vib_w = vib[s:e]
            temp_w = self._slice_temp_context(temp, s, e)
            x = self.transform(vib_w) if self.transform else vib_w
            if self.temp_feature_fn:
                t = self.temp_feature_fn(temp_w)
            else:
                t = np.zeros(self.temp_feat_dim, dtype=np.float32)
            X.append(x)
            T.append(t)
        if not X:
            raise IndexError(f"No windows for file: {item['path']}")
        X = torch.stack(X, dim=0)  # (W,2,H,W)
        T = torch.tensor(np.stack(T, axis=0), dtype=torch.float32)  # (W,D)
        y = torch.tensor(item["label"], dtype=torch.long)
        return X, T, y
