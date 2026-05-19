import json
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

ROWS_PER_BAR  = 192
ROWS_PER_BEAT = 48       # ROWS_PER_BAR // 4
WINDOW_BARS   = 4
WINDOW_ROWS   = WINDOW_BARS * ROWS_PER_BAR   # 768
NUM_PATCHES   = WINDOW_ROWS // ROWS_PER_BEAT  # 16  (one patch per beat)
PATCH_ROWS    = ROWS_PER_BEAT                 # 48
STRIDE_BARS   = 2
STRIDE_ROWS   = STRIDE_BARS * ROWS_PER_BAR   # 384
IN_CHANNELS   = 33  # 16 lanes × 2 + 1 BPM

HEAD_TYPES = frozenset({1, 2, 4, 6, 8})           # tap, CN/HCN/BSS/MSS heads
BODY_SUSTAIN = {3: 1.0, 5: 2.0, 7: 0.5, 9: 0.5}  # CN, HCN, BSS/MSS bodies

GREAT_WINDOW_SEC = 0.033  # ±33ms timing window used for Gaussian smear σ

# Lane permutations: PERM[i] = source lane for output lane i
MIRROR_PERM         = [0, 7, 6, 5, 4, 3, 2, 1, 14, 13, 12, 11, 10, 9, 8, 15]
SIDESWAP_PERM       = [15, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 0]
MIRROR_SIDESWAP_PERM = [SIDESWAP_PERM[MIRROR_PERM[i]] for i in range(16)]


# ── Low-level encoding ────────────────────────────────────────────────────────

def _encode_lanes(lanes: np.ndarray) -> np.ndarray:
    """
    lanes: (total_rows, 16) uint32
    Returns: (total_rows, 16, 2) float32
      channel 0 = action  (1.0 at heads and hold tails)
      channel 1 = sustain (hold body magnitude)
    """
    total_rows = lanes.shape[0]
    action  = np.zeros((total_rows, 16), dtype=np.float32)
    sustain = np.zeros((total_rows, 16), dtype=np.float32)

    for lane_idx in range(16):
        col = lanes[:, lane_idx]
        for r in range(total_rows):
            v = int(col[r])
            if v in HEAD_TYPES:
                action[r, lane_idx] = 1.0
            elif v in BODY_SUSTAIN:
                sustain[r, lane_idx] = BODY_SUSTAIN[v]
                # tail: last row of this body run
                if r + 1 == total_rows or int(col[r + 1]) != v:
                    action[r, lane_idx] = 1.0

    return np.stack([action, sustain], axis=2)  # (total_rows, 16, 2)


def _smear_action(action: np.ndarray, sigma_rows: float) -> np.ndarray:
    """Gaussian blur along the row axis (axis 0) for all lanes simultaneously."""
    sigma_rows = max(0.5, sigma_rows)
    radius = max(1, int(3.0 * sigma_rows + 0.5))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (x / sigma_rows) ** 2)
    kernel /= kernel.sum()

    total_rows = action.shape[0]
    out = np.zeros_like(action)
    for i, w in enumerate(kernel):
        shift = i - radius
        src_s = max(0, shift);   src_e = min(total_rows, total_rows + shift)
        dst_s = max(0, -shift);  dst_e = min(total_rows, total_rows - shift)
        out[dst_s:dst_e] += w * action[src_s:src_e]
    return out


def encode_chart(npy_path) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load and encode a single chart.

    Returns
    -------
    encoded : (total_rows, 33) float32
        16 lanes × 2 channels (action smeared, sustain) + 1 BPM channel.
    note_counts_row : (total_rows,) int32
        Number of note-head events (across all lanes) at each row.
    meta : dict
    """
    npy_path = Path(npy_path)
    arr  = np.load(npy_path)                                   # (total_rows, 17) uint32
    meta = json.loads(npy_path.with_suffix('.json').read_text())

    lanes   = arr[:, :16]                                      # (total_rows, 16) uint32
    bpm_col = arr[:, 16].astype(np.float32) / 100.0           # BPM (dense, fill-forwarded)

    encoded_lanes = _encode_lanes(lanes)                       # (total_rows, 16, 2)

    mean_bpm   = float(bpm_col.mean())
    sigma_rows = GREAT_WINDOW_SEC * (mean_bpm / 60.0) * ROWS_PER_BEAT

    action_smeared = _smear_action(encoded_lanes[:, :, 0], sigma_rows)
    encoded_lanes[:, :, 0] = action_smeared

    flat    = encoded_lanes.reshape(arr.shape[0], 32)          # (total_rows, 32)
    encoded = np.concatenate([flat, bpm_col[:, None]], axis=1) # (total_rows, 33)

    note_counts_row = np.isin(lanes, list(HEAD_TYPES)).sum(axis=1).astype(np.int32)

    return encoded, note_counts_row, meta


def window_chart(
    encoded: np.ndarray,
    note_counts_row: np.ndarray,
    window_bars: int = WINDOW_BARS,
    stride_bars: int = STRIDE_BARS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slice a chart into overlapping fixed-length windows.

    Returns
    -------
    windows : (num_windows, window_rows, 33) float32
    win_note_counts : (num_windows,) int32  — total note events per window
    """
    wr = window_bars * ROWS_PER_BAR
    sr = stride_bars * ROWS_PER_BAR
    total = encoded.shape[0]

    starts = range(0, total - wr + 1, sr)
    if not starts:
        return (np.zeros((0, wr, IN_CHANNELS), dtype=np.float32),
                np.zeros(0, dtype=np.int32))

    windows = np.stack([encoded[s:s + wr] for s in starts])
    counts  = np.array([int(note_counts_row[s:s + wr].sum()) for s in starts],
                       dtype=np.int32)
    return windows, counts


# ── Augmentation ─────────────────────────────────────────────────────────────

def apply_lane_perm(windows: np.ndarray, perm) -> np.ndarray:
    """
    Rearrange lane channels in encoded windows.

    windows : (..., 33)  last dim = [lane0_ch0, lane0_ch1, ..., lane15_ch0, lane15_ch1, bpm]
    perm    : list of 16 ints — source lane for each output lane position
    """
    col_idx = [c for lane in perm for c in (2 * lane, 2 * lane + 1)] + [32]
    return windows[..., col_idx]


# ── Datasets ─────────────────────────────────────────────────────────────────

class PretrainDataset(Dataset):
    """Individual 4-bar windows from all levels — used for MAE pretraining."""

    def __init__(
        self,
        manifest_csv: str,
        data_root: str,
        augment: bool = True,
        window_bars: int = WINDOW_BARS,
        stride_bars: int = STRIDE_BARS,
    ):
        self.data_root   = Path(data_root)
        self.augment     = augment
        self.window_bars = window_bars
        self.stride_bars = stride_bars

        manifest = pd.read_csv(manifest_csv)
        manifest = manifest[manifest['status'] == 'ok'].reset_index(drop=True)

        self.level_dir = {10: 'dp10_active', 11: 'dp11_active', 12: 'dp12_active'}

        # Build flat index: list of (npy_path, window_start_row)
        self.index: list[tuple[Path, int]] = []
        wr = window_bars * ROWS_PER_BAR
        sr = stride_bars * ROWS_PER_BAR

        for _, row in manifest.iterrows():
            level_dir = self.data_root / self.level_dir[int(row['level'])]
            npy = level_dir / str(row['file_path'])
            if not npy.exists():
                continue
            meta = json.loads(npy.with_suffix('.json').read_text())
            total_rows = meta['total_bars'] * ROWS_PER_BAR
            for start in range(0, total_rows - wr + 1, sr):
                self.index.append((npy, start))

        self._aug_perms = [MIRROR_PERM, SIDESWAP_PERM, MIRROR_SIDESWAP_PERM]

    def __len__(self):
        base = len(self.index)
        return base * 4 if self.augment else base

    def __getitem__(self, idx):
        aug_id  = idx % 4 if self.augment else 0
        base_idx = idx // 4 if self.augment else idx

        npy_path, start = self.index[base_idx]
        encoded, _, _ = encode_chart(npy_path)
        wr = self.window_bars * ROWS_PER_BAR
        window = encoded[start:start + wr]  # (768, 33)

        if aug_id > 0:
            perm   = self._aug_perms[aug_id - 1]
            window = apply_lane_perm(window, perm)

        return torch.from_numpy(window)  # (768, 33)


class FinetuneDataset(Dataset):
    """
    One item = one lv12 chart with all its windows and gauge ratings.
    Use with batch_size=1; the gauge simulation is sequential over windows.
    """

    def __init__(
        self,
        manifest_csv: str,
        data_root: str,
        augment: bool = True,
        window_bars: int = WINDOW_BARS,
        stride_bars: int = STRIDE_BARS,
    ):
        self.data_root   = Path(data_root)
        self.augment     = augment
        self.window_bars = window_bars
        self.stride_bars = stride_bars

        manifest = pd.read_csv(manifest_csv)
        # lv12 charts with all three gauge ratings
        lv12 = manifest[
            (manifest['level'] == 12) &
            manifest['rating_ec'].notna() &
            manifest['rating_hc'].notna() &
            manifest['rating_exh'].notna()
        ].copy().reset_index(drop=True)

        self.entries = []
        for _, row in lv12.iterrows():
            npy = self.data_root / 'dp12_active' / str(row['file_path'])
            if npy.exists():
                self.entries.append({
                    'npy': npy,
                    'rating_ec':  float(row['rating_ec']),
                    'rating_hc':  float(row['rating_hc']),
                    'rating_exh': float(row['rating_exh']),
                    'rating_stat': float(row['rating_stat']) if pd.notna(row.get('rating_stat')) else None,
                })

        self._aug_perms = [None, MIRROR_PERM, SIDESWAP_PERM, MIRROR_SIDESWAP_PERM]

    def __len__(self):
        return len(self.entries) * (4 if self.augment else 1)

    def __getitem__(self, idx):
        aug_id   = idx % 4 if self.augment else 0
        base_idx = idx // 4 if self.augment else idx

        entry = self.entries[base_idx]
        encoded, note_counts_row, _ = encode_chart(entry['npy'])
        windows, win_note_counts = window_chart(encoded, note_counts_row,
                                                self.window_bars, self.stride_bars)

        perm = self._aug_perms[aug_id]
        if perm is not None:
            windows = apply_lane_perm(windows, perm)

        ratings = torch.tensor([
            entry['rating_ec'],
            entry['rating_hc'],
            entry['rating_exh'],
        ], dtype=torch.float32)

        return (
            torch.from_numpy(windows),         # (T, 768, 33)
            torch.from_numpy(win_note_counts), # (T,) int32
            ratings,                           # (3,)
        )
