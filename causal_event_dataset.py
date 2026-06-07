"""
PyTorch Dataset for causal next-token prediction on per-timestep event tokens.

Each item is a fixed-token-count window. The model sees tokens 0..T-1 (with
causal attention) and predicts tokens 1..T.

Fixed-token windowing
---------------------
Windows slide by token index (not by time). Every window contains exactly
WIN_TOKENS real tokens — no padding. This eliminates the positional bias from
the previous time-based scheme, where sparse windows had their last-token
embedding at a much lower position index than dense windows.

Temporal information is preserved through delta_bins: the note-speed
quantisation encodes how far apart events are in real time, so BPM
information is not lost despite the time-agnostic window boundaries.

Targets (shifted by one):
    tgt_lane_bits[t] = lane_bits[t+1]   for t in 0..T-2
    tgt_delta[t]     = delta_bins[t+1]  for t in 0..T-2, else -100
    tgt_ntype[t]     = note_types[t+1]  for t in 0..T-2, else -100
is_causal_valid[t] = True for t in 0..T-2

Augmentations (same as before):
  side_swap, mirror, both — 4× dataset
"""

import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset

from event_tokenize import quantize_delta_vec, DELTA_BINS
from event_dataset   import (
    NUM_LANES, NUM_NTYPES, masks_to_bits,
    _permute_masks, _SIDE_SWAP, _MIRROR,
)

WIN_TOKENS    = 48
STRIDE_TOKENS = 24
MAX_TOKENS    = WIN_TOKENS   # alias used by encoder for pos_embed size
MIN_CHART_TOKENS = WIN_TOKENS


class CausalEventDataset(Dataset):
    """
    Fixed-token sliding-window dataset for causal next-token prediction.

    Parameters
    ----------
    cache_dir      : directory of .npz files produced by event_tokenize.py
    win_tokens     : tokens per window (every window is exactly this size)
    stride_tokens  : token-count stride between window starts
    augment        : if True, add side-swap + mirror variants (4× dataset)
    """

    def __init__(
        self,
        cache_dir:     str,
        win_tokens:    int  = WIN_TOKENS,
        stride_tokens: int  = STRIDE_TOKENS,
        augment:       bool = True,
    ):
        self.win_tokens = win_tokens

        self._windows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        cache_dir_p = Path(cache_dir).expanduser()
        npz_paths   = sorted(cache_dir_p.glob('*.npz'))
        if not npz_paths:
            raise FileNotFoundError(f'No .npz files found in {cache_dir_p}')

        for npz_path in npz_paths:
            data = np.load(npz_path)
            lm   = data['lane_masks']
            nt   = data['note_types']
            ts   = data['times_sec']

            n = len(ts)
            if n < win_tokens:
                continue

            t0_idx = 0
            while t0_idx + win_tokens <= n:
                idx  = np.arange(t0_idx, t0_idx + win_tokens)
                lm_w = lm[idx]
                nt_w = nt[idx]
                ts_w = ts[idx]

                self._windows.append((lm_w, nt_w, ts_w))
                if augment:
                    self._windows.append(
                        (_permute_masks(lm_w, _SIDE_SWAP), nt_w, ts_w))
                    self._windows.append(
                        (_permute_masks(lm_w, _MIRROR), nt_w, ts_w))
                    self._windows.append(
                        (_permute_masks(_permute_masks(lm_w, _SIDE_SWAP),
                                        _MIRROR), nt_w, ts_w))
                t0_idx += stride_tokens

        print(f'CausalEventDataset: {len(self._windows)} windows '
              f'from {len(npz_paths)} charts  '
              f'(win={win_tokens} toks, stride={stride_tokens} toks, augment={augment})')

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        lm, nt, ts = self._windows[idx]
        T = self.win_tokens  # always exactly win_tokens, no padding

        deltas = np.zeros(T, dtype=np.float64)
        deltas[1:] = np.diff(ts.astype(np.float64))
        db     = quantize_delta_vec(deltas).astype(np.int64)

        lane_bits = masks_to_bits(lm)   # (T, 16)
        nt_arr    = nt.astype(np.int64)

        # No padding — every window is full.
        is_padded = np.zeros(T, dtype=bool)

        # Position T-1 has no next token; all others are valid prediction targets.
        is_causal_valid        = np.zeros(T, dtype=bool)
        is_causal_valid[:T-1]  = True

        tgt_lane_bits          = np.zeros((T, NUM_LANES), dtype=np.float32)
        tgt_delta              = np.full(T, -100, dtype=np.int64)
        tgt_ntype              = np.full(T, -100, dtype=np.int64)

        tgt_lane_bits[:T-1] = lane_bits[1:T]
        tgt_delta[:T-1]     = db[1:T]
        tgt_ntype[:T-1]     = nt_arr[1:T]

        return {
            'lane_bits':       torch.from_numpy(lane_bits),       # (T, 16)
            'delta_bins':      torch.from_numpy(db),              # (T,)
            'note_types':      torch.from_numpy(nt_arr),          # (T,)
            'is_padded':       torch.from_numpy(is_padded),       # (T,) — always False
            'is_causal_valid': torch.from_numpy(is_causal_valid), # (T,)
            'tgt_lane_bits':   torch.from_numpy(tgt_lane_bits),   # (T, 16)
            'tgt_delta':       torch.from_numpy(tgt_delta),       # (T,)
            'tgt_ntype':       torch.from_numpy(tgt_ntype),       # (T,)
        }
