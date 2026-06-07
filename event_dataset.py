"""
PyTorch Dataset for MAE pretraining on per-timestep event token sequences.

Each item is a fixed-length window of event tokens with a random subset masked.
The model sees the original values everywhere and must predict the masked positions.

Token fields (per timestep)
---------------------------
  lane_bits  : (16,) float32  multi-hot binary — which lanes are active
  delta_bin  : int            log-quantised ms since previous timestep in window
                              0 = first token in window
  note_type  : int            0=tap  1=cn  2=hcn  3=bss/mss

Masking
-------
  is_masked positions: model input is replaced with a learned mask token (done
  inside the model, not here).  Dataset returns is_masked and per-field targets:
    tgt_lane_bits : (MAX_TOKENS, 16) float32  — original values at ALL positions
    tgt_delta     : (MAX_TOKENS,)   int64     — original delta, -100 elsewhere
    tgt_ntype     : (MAX_TOKENS,)   int64     — original note_type, -100 elsewhere

Augmentations
-------------
  side_swap : P1 ↔ P2 (lane i ↔ lane 15-i), mirrors the full 16-lane strip
  mirror_p1 : flip keys within P1 side (lanes 1↔7, 2↔6, 3↔5)
  mirror_p2 : flip keys within P2 side (lanes 8↔14, 9↔13, 10↔12)
  All three are independent booleans; combining gives up to 8× dataset.
"""

import math
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from event_tokenize import quantize_delta_vec, DELTA_BINS

# ── constants ─────────────────────────────────────────────────────────────────

WIN_SEC    = 4.0
STRIDE_SEC = 2.0
MAX_TOKENS = 128
MIN_TOKENS = 8
MASK_RATE  = 0.30
NUM_LANES  = 16
NUM_NTYPES = 4


# ── lane permutations ─────────────────────────────────────────────────────────

# Side-swap: lane i → lane 15-i
_SIDE_SWAP = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
                       dtype=np.int32)

# Within-side mirror: scratch stays, keys 1-7 reversed on each side
# P1: lanes 0(scr) 1 2 3 4 5 6 7  →  0 7 6 5 4 3 2 1
# P2: lanes 15(scr) 14 13 12 11 10 9 8  →  same permutation on global indices
_MIRROR = np.array([0,  7,  6,  5,  4,  3,  2,  1,
                    15, 14, 13, 12, 11, 10,  9,  8], dtype=np.int32)


def _permute_masks(masks: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Apply a lane permutation to an array of uint16 bitmasks.

    perm[i] = j means output lane i comes from input lane j.
    """
    bits = ((masks[:, None].astype(np.int32) >> np.arange(16)) & 1)  # (N, 16)
    permuted = bits[:, perm]                                           # (N, 16)
    powers   = (1 << np.arange(16, dtype=np.int32))
    return (permuted * powers).sum(axis=1).astype(np.uint16)


# ── bit expansion ─────────────────────────────────────────────────────────────

def masks_to_bits(masks: np.ndarray) -> np.ndarray:
    """(N,) uint16 → (N, 16) float32 multi-hot binary."""
    return ((masks[:, None].astype(np.uint16) >> np.uint16(np.arange(16))) & 1
            ).astype(np.float32)


# ── dataset ───────────────────────────────────────────────────────────────────

class EventMAEDataset(Dataset):
    """
    Sliding-window dataset over event token caches produced by event_tokenize.py.

    Parameters
    ----------
    cache_dir   : directory of .npz files (one per chart)
    win_sec     : physical seconds per window
    stride_sec  : stride between window starts
    max_tokens  : pad / truncate each window to this length
    min_tokens  : skip windows with fewer active tokens
    mask_rate   : fraction of non-padded positions to mask
    augment     : if True, add side-swap + mirror variants (up to 4× dataset)
    """

    def __init__(
        self,
        cache_dir:  str,
        win_sec:    float = WIN_SEC,
        stride_sec: float = STRIDE_SEC,
        max_tokens: int   = MAX_TOKENS,
        min_tokens: int   = MIN_TOKENS,
        mask_rate:  float = MASK_RATE,
        augment:    bool  = True,
    ):
        self.max_tokens = max_tokens
        self.mask_rate  = mask_rate

        # Each entry: (lane_masks, note_types, times_sec) for one window
        # lane_masks / note_types are raw (possibly permuted); delta is recomputed
        self._windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        cache_dir_p = Path(cache_dir).expanduser()
        npz_paths   = sorted(cache_dir_p.glob('*.npz'))
        if not npz_paths:
            raise FileNotFoundError(f'No .npz files found in {cache_dir_p}')

        for npz_path in npz_paths:
            data = np.load(npz_path)
            lm   = data['lane_masks']   # (N,) uint16
            nt   = data['note_types']   # (N,) uint8
            ts   = data['times_sec']    # (N,) float32

            if len(ts) < min_tokens:
                continue

            t_max = float(ts[-1])
            t0    = 0.0
            while t0 < t_max:
                idx = np.where((ts >= t0) & (ts < t0 + win_sec))[0]
                if len(idx) >= min_tokens:
                    idx = idx[:max_tokens]
                    base = (lm[idx], nt[idx], ts[idx])
                    self._windows.append(base)
                    if augment:
                        # side-swap
                        self._windows.append(
                            (_permute_masks(lm[idx], _SIDE_SWAP), nt[idx], ts[idx]))
                        # within-side mirror
                        self._windows.append(
                            (_permute_masks(lm[idx], _MIRROR), nt[idx], ts[idx]))
                        # side-swap + mirror
                        self._windows.append(
                            (_permute_masks(_permute_masks(lm[idx], _SIDE_SWAP),
                                            _MIRROR), nt[idx], ts[idx]))
                t0 += stride_sec

        print(f'EventMAEDataset: {len(self._windows)} windows '
              f'from {len(npz_paths)} charts  '
              f'(augment={augment})')

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        lm, nt, ts = self._windows[idx]
        T = len(lm)

        # Recompute deltas within this window (bin 0 for first token)
        deltas = np.zeros(T, dtype=np.float64)
        if T > 1:
            deltas[1:] = np.diff(ts.astype(np.float64))
        db = quantize_delta_vec(deltas).astype(np.int64)

        # Expand lane masks to multi-hot float
        lane_bits = masks_to_bits(lm)         # (T, 16)

        # Pad to max_tokens
        pad = self.max_tokens - T
        if pad > 0:
            lane_bits = np.pad(lane_bits, ((0, pad), (0, 0)))
            db        = np.pad(db,        (0, pad))
            nt_pad    = np.pad(nt.astype(np.int64), (0, pad))
        else:
            nt_pad = nt.astype(np.int64)

        is_padded       = np.zeros(self.max_tokens, dtype=bool)
        is_padded[T:]   = True

        # Random masking of non-padded positions
        real_pos = np.where(~is_padded)[0]
        n_mask   = max(1, int(len(real_pos) * self.mask_rate))
        mask_pos = np.random.choice(real_pos, size=n_mask, replace=False)
        is_masked = np.zeros(self.max_tokens, dtype=bool)
        is_masked[mask_pos] = True

        # Targets: keep original values at masked positions, -100 elsewhere
        tgt_delta = np.full(self.max_tokens, -100, dtype=np.int64)
        tgt_ntype = np.full(self.max_tokens, -100, dtype=np.int64)
        tgt_delta[mask_pos] = db[mask_pos]
        tgt_ntype[mask_pos] = nt_pad[mask_pos]

        return {
            # inputs — model replaces is_masked positions with its mask token
            'lane_bits':     torch.from_numpy(lane_bits),         # (T, 16)
            'delta_bins':    torch.from_numpy(db),                # (T,)
            'note_types':    torch.from_numpy(nt_pad),            # (T,)
            'is_masked':     torch.from_numpy(is_masked),         # (T,) bool
            'is_padded':     torch.from_numpy(is_padded),         # (T,) bool
            # targets
            'tgt_lane_bits': torch.from_numpy(lane_bits),         # (T, 16) full
            'tgt_delta':     torch.from_numpy(tgt_delta),         # (T,) -100 elsewhere
            'tgt_ntype':     torch.from_numpy(tgt_ntype),         # (T,) -100 elsewhere
        }
