import json
import functools
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
IN_CHANNELS              = 33  # 16 lanes × 2 + 1 BPM  (fine-tuning)
PRETRAIN_IN_CHANNELS     = 32  # BPM dropped for pretraining (v2)
PRETRAIN_IN_CHANNELS_V3  = 16  # single channel per lane (v3/v4)
PRETRAIN_IN_CHANNELS_V6  = 16  # v6: pure binary, no BPM in channels (same value, distinct name)
PATCH_ROWS_V4            = 12  # sub-beat (16th-note) resolution
NUM_PATCHES_V4           = WINDOW_ROWS // PATCH_ROWS_V4  # 64

# v7: BPM-agnostic time-based windowing (8 seconds, 150 BPM reference)
REF_BPM          = 150.0
WIN_SECS         = 8.0
STRIDE_SECS      = 4.0
WIN_ROWS_V7      = int(WIN_SECS * REF_BPM / 60.0 * ROWS_PER_BAR / 4)  # 960
ROWS_PER_SEC_REF = REF_BPM * ROWS_PER_BAR / (4.0 * 60.0)              # 120.0
PATCH_ROWS_V7    = PATCH_ROWS_V4   # 12 rows = 16th-note resolution
NUM_PATCHES_V7   = WIN_ROWS_V7 // PATCH_ROWS_V7  # 80

# v10: 4-second windows, 2-second stride (half the v8/v9 window)
WIN_SECS_V10    = 4.0
STRIDE_SECS_V10 = 2.0
WIN_ROWS_V10    = int(WIN_SECS_V10 * REF_BPM / 60.0 * ROWS_PER_BAR / 4)  # 480
NUM_PATCHES_V10 = WIN_ROWS_V10 // PATCH_ROWS_V7  # 40

# v12: same 4s windows but 4-row patches (~33ms each at 150 BPM ref = 120 rows/sec)
# Resolves 32nd notes at 150 BPM (50ms) and 16th notes up to ~300 BPM
PATCH_ROWS_V12  = 4
NUM_PATCHES_V12 = WIN_ROWS_V10 // PATCH_ROWS_V12  # 120

# Hand-split constants.
# P1 format: [scratch(0), key1(1)..key7(7)]  — P1 is already in this order.
# P2 format in raw array: [key1(8)..key7(14), scratch(15)] — reorder to P1 format.
P1_LANE_COLS             = [0, 1, 2, 3, 4, 5, 6, 7]
P2_TO_P1_LANE_COLS       = [15, 8, 9, 10, 11, 12, 13, 14]  # scratch first, then key1–7
PRETRAIN_IN_CHANNELS_HALF = 8   # 8 lanes (one hand), no BPM channel
IN_CHANNELS_HALF          = 9   # 8 lanes + 1 BPM

# Within a single hand: mirror flips keys 1–7, scratch (position 0) stays fixed.
MIRROR_PERM_HALF         = [0, 7, 6, 5, 4, 3, 2, 1]

# Lane-type indices for the half-chart (P1 format): scratch=0, keys=1.
LANE_TYPES_HALF          = [0, 1, 1, 1, 1, 1, 1, 1]

HEAD_TYPES = frozenset({1, 2, 4, 6, 8})           # tap, CN/HCN/BSS/MSS heads
BODY_SUSTAIN = {3: 1.0, 5: 2.0, 7: 0.5, 9: 0.5}  # CN, HCN, BSS/MSS bodies

GREAT_WINDOW_SEC = 0.033  # ±33ms timing window used for Gaussian smear σ
BPM_SCALE = 250.0         # normalises BPM channel to ~0–1 range

# Pretrain-only fat-binary encoding — asymmetric by lane type.
#
# Key lanes (1–14): h=5 → 11 rows total
#   - Common dense case is 16th notes (12-row gap); h=5 leaves 1-row gap → near-touching, game-accurate
#   - At 180 BPM: ±8.7ms ≈ half the PGREAT window
#
# Scratch lanes (0, 15): h=3 → 7 rows total
#   - Scratch p5 minimum gap = 7 rows (dense burst walls); h=3 leaves 0-row gap (just touching) in 1% of windows
#   - h=5 would cause 4-row overlap in 4.6% of scratch windows — structural confusion in burst sections
#   - Thinner scratch matches the game's visual: scratch is a distinct disc object, not a rectangular key bar
PRETRAIN_KEY_HALF_WIDTH     = 5   # lanes 1–14
PRETRAIN_SCRATCH_HALF_WIDTH = 3   # lanes 0, 15
SCRATCH_LANES = frozenset({0, 15})

# Lane permutations: PERM[i] = source lane for output lane i.
# Two generators: mirror (reverse key order within a side, scratch fixed) and
# side-swap (exchange P1 and P2 entirely including scratches). Their 8
# combinations (2^3 independent bits: mirror-P1, mirror-P2, side-swap) give the
# full augmentation group. _compose_perm(a, b)[i] = a[b[i]] → "apply a then b".
def _compose_perm(a, b):
    return [a[b[i]] for i in range(16)]

MIRROR_P1_PERM            = [0, 7, 6, 5, 4, 3, 2, 1,  8,  9, 10, 11, 12, 13, 14, 15]
MIRROR_P2_PERM            = [0, 1, 2, 3, 4, 5, 6, 7, 14, 13, 12, 11, 10,  9,  8, 15]
MIRROR_PERM               = _compose_perm(MIRROR_P1_PERM, MIRROR_P2_PERM)
SIDESWAP_PERM             = [15, 8, 9, 10, 11, 12, 13, 14, 1, 2, 3, 4, 5, 6, 7, 0]
SIDESWAP_MIRROR_P1_PERM   = _compose_perm(SIDESWAP_PERM, MIRROR_P1_PERM)
SIDESWAP_MIRROR_P2_PERM   = _compose_perm(SIDESWAP_PERM, MIRROR_P2_PERM)
SIDESWAP_MIRROR_BOTH_PERM = _compose_perm(SIDESWAP_PERM, MIRROR_PERM)

_ALL_AUG_PERMS = [
    MIRROR_P1_PERM,
    MIRROR_P2_PERM,
    MIRROR_PERM,
    SIDESWAP_PERM,
    SIDESWAP_MIRROR_P1_PERM,
    SIDESWAP_MIRROR_P2_PERM,
    SIDESWAP_MIRROR_BOTH_PERM,
]  # 7 non-identity permutations → 8× dataset with identity


# ── Low-level encoding ────────────────────────────────────────────────────────

def _encode_lanes(lanes: np.ndarray) -> np.ndarray:
    """
    lanes: (total_rows, 16) uint32
    Returns: (total_rows, 16, 2) float32
      channel 0 = action  (1.0 at heads and hold tails)
      channel 1 = sustain (hold body magnitude)
    """
    action  = np.zeros(lanes.shape, dtype=np.float32)
    sustain = np.zeros(lanes.shape, dtype=np.float32)

    # heads → action = 1
    head_arr = np.array(sorted(HEAD_TYPES), dtype=lanes.dtype)
    action[np.isin(lanes, head_arr)] = 1.0

    # body types → sustain value + action=1 at run tail
    for v, s in BODY_SUSTAIN.items():
        body = (lanes == v)                            # (total_rows, 16) bool
        sustain[body] = s
        # tail row: body is True here but False on the next row (or last row)
        tail = body & ~np.concatenate([body[1:], np.zeros((1, 16), dtype=bool)], axis=0)
        action[tail] = 1.0

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


@functools.lru_cache(maxsize=512)
def encode_chart(npy_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Load and encode a single chart.  Results are cached per worker process.

    Returns
    -------
    encoded : (total_rows, 33) float32
        16 lanes × 2 channels (action smeared, sustain) + 1 BPM channel (normalised).
    note_counts_row : (total_rows,) int32
        Number of note-head events (across all lanes) at each row.
    meta : dict
    """
    npy_path = Path(npy_path)
    arr  = np.load(npy_path)                                         # (total_rows, 17) uint32
    meta = json.loads(npy_path.with_suffix('.json').read_text())

    lanes   = arr[:, :16]                                            # (total_rows, 16) uint32
    bpm_raw = arr[:, 16].astype(np.float32)
    # col 16 is sparse: non-zero only at BPM-change rows; fill forward
    mask = bpm_raw != 0
    idx  = np.where(mask, np.arange(len(bpm_raw)), 0)
    np.maximum.accumulate(idx, out=idx)
    bpm_col = bpm_raw[idx] / 100.0                                   # BPM (filled), normalised later

    encoded_lanes = _encode_lanes(lanes)                             # (total_rows, 16, 2)

    mean_bpm   = float(bpm_col.mean())
    sigma_rows = GREAT_WINDOW_SEC * (mean_bpm / 60.0) * ROWS_PER_BEAT

    action_smeared = _smear_action(encoded_lanes[:, :, 0], sigma_rows)
    encoded_lanes[:, :, 0] = action_smeared

    flat    = encoded_lanes.reshape(arr.shape[0], 32)                # (total_rows, 32)
    bpm_norm = bpm_col / BPM_SCALE                                   # normalise ~0–1
    encoded = np.concatenate([flat, bpm_norm[:, None]], axis=1)      # (total_rows, 33)

    note_counts_row = np.isin(lanes, list(HEAD_TYPES)).sum(axis=1).astype(np.int32)

    return encoded, note_counts_row, meta


def _encode_lanes_pretrain(lanes: np.ndarray) -> np.ndarray:
    """
    Simplified two-type encoding for MAE pretraining.

    Channel 0 — tap: binary 1.0 box around every note head.
                Key lanes (1–14): ±PRETRAIN_KEY_HALF_WIDTH rows (11 rows total).
                Scratch lanes (0, 15): ±PRETRAIN_SCRATCH_HALF_WIDTH rows (7 rows total).
    Channel 1 — hold: binary 1.0 during the body of CN/HCN/BSS/MSS (head through tail).

    Returns: (total_rows, 16, 2) float32
    """
    total_rows = lanes.shape[0]
    tap  = np.zeros((total_rows, 16), dtype=np.float32)
    hold = np.zeros((total_rows, 16), dtype=np.float32)

    head_arr  = np.array(sorted(HEAD_TYPES), dtype=lanes.dtype)
    head_mask = np.isin(lanes, head_arr)                        # (total_rows, 16) bool

    key_kern = np.ones(2 * PRETRAIN_KEY_HALF_WIDTH + 1,     dtype=np.float32)
    sc_kern  = np.ones(2 * PRETRAIN_SCRATCH_HALF_WIDTH + 1, dtype=np.float32)
    for lane in range(16):
        kern = sc_kern if lane in SCRATCH_LANES else key_kern
        tap[:, lane] = (np.convolve(head_mask[:, lane].astype(np.float32), kern, mode='same') > 0).astype(np.float32)

    # hold body
    body_arr  = np.array(sorted(BODY_SUSTAIN.keys()), dtype=lanes.dtype)
    body_mask = np.isin(lanes, body_arr)
    hold[body_mask] = 1.0
    hold_heads = np.isin(lanes, np.array([2, 4, 6, 8], dtype=lanes.dtype))
    hold[head_mask & hold_heads] = 1.0

    return np.stack([tap, hold], axis=2)                        # (total_rows, 16, 2)


@functools.lru_cache(maxsize=512)
def encode_chart_pretrain(npy_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Fat-binary two-type encoding for MAE pretraining. Cached per worker process.

    Returns
    -------
    encoded : (total_rows, 33) float32
        16 lanes × 2 channels (fat tap binary, hold binary) + 1 BPM channel (normalised).
    note_counts_row : (total_rows,) int32
    meta : dict
    """
    npy_path = Path(npy_path)
    arr  = np.load(npy_path)
    meta = json.loads(npy_path.with_suffix('.json').read_text())

    lanes   = arr[:, :16]
    bpm_raw = arr[:, 16].astype(np.float32)
    mask = bpm_raw != 0
    idx  = np.where(mask, np.arange(len(bpm_raw)), 0)
    np.maximum.accumulate(idx, out=idx)
    bpm_col = bpm_raw[idx] / 100.0

    encoded_lanes = _encode_lanes_pretrain(lanes)          # (total_rows, 16, 2)

    flat     = encoded_lanes.reshape(arr.shape[0], 32)
    bpm_norm = bpm_col / BPM_SCALE
    encoded  = np.concatenate([flat, bpm_norm[:, None]], axis=1)   # (total_rows, 33)

    note_counts_row = np.isin(lanes, list(HEAD_TYPES)).sum(axis=1).astype(np.int32)

    return encoded, note_counts_row, meta


def _encode_lanes_v3(lanes: np.ndarray) -> np.ndarray:
    """
    Single-channel encoding: 0.0=empty, 1.0=tap, 0.5=hold body.
    Key lanes (1-14): raw binary — exact note rows only, no smear.
    Scratch lanes (0, 15): ±1 row smear to absorb quantisation jitter.
    Tap always overrides hold body in overlap.
    Returns: (total_rows, 16) float32
    """
    total_rows = lanes.shape[0]
    out = np.zeros((total_rows, 16), dtype=np.float32)

    head_arr = np.array(sorted(HEAD_TYPES), dtype=lanes.dtype)
    head_mask = np.isin(lanes, head_arr)  # (total_rows, 16) bool

    body_arr = np.array(sorted(BODY_SUSTAIN.keys()), dtype=lanes.dtype)
    out[np.isin(lanes, body_arr)] = 0.5   # step 1: hold bodies → 0.5

    sc_kern = np.ones(3, dtype=np.float32)  # ±1 row for scratch quantisation jitter
    for lane in range(16):
        if lane in SCRATCH_LANES:
            fired = np.convolve(head_mask[:, lane].astype(np.float32), sc_kern, mode='same') > 0
        else:
            fired = head_mask[:, lane]
        out[:, lane] = np.where(fired, 1.0, out[:, lane])  # step 2: tap → 1.0

    return out


@functools.lru_cache(maxsize=512)
def encode_chart_pretrain_v3(npy_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Single-channel v3 encoding for MAE pretraining.

    Returns
    -------
    encoded : (total_rows, 17) float32
        16 lane channels (v3) + BPM at col 16 (normalised by BPM_SCALE).
    note_counts_row : (total_rows,) int32
    meta : dict
    """
    npy_path = Path(npy_path)
    arr  = np.load(npy_path)
    meta = json.loads(npy_path.with_suffix('.json').read_text())

    lanes   = arr[:, :16]
    bpm_raw = arr[:, 16].astype(np.float32)
    mask = bpm_raw != 0
    idx  = np.where(mask, np.arange(len(bpm_raw)), 0)
    np.maximum.accumulate(idx, out=idx)
    bpm_col = bpm_raw[idx] / 100.0

    encoded_lanes = _encode_lanes_v3(lanes)                            # (total_rows, 16)
    bpm_norm      = bpm_col / BPM_SCALE
    encoded       = np.concatenate([encoded_lanes, bpm_norm[:, None]], axis=1)  # (total_rows, 17)

    note_counts_row = np.isin(lanes, list(HEAD_TYPES)).sum(axis=1).astype(np.int32)
    return encoded, note_counts_row, meta


def _encode_lanes_v6(lanes: np.ndarray) -> np.ndarray:
    """Pure binary: 1.0 at head notes, 0.0 everywhere else. No hold bodies, no smear."""
    head_arr = np.array(sorted(HEAD_TYPES), dtype=lanes.dtype)
    return np.isin(lanes, head_arr).astype(np.float32)  # (total_rows, 16)


@functools.lru_cache(maxsize=512)
def encode_chart_pretrain_v6(npy_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Pure binary v6 encoding for MAE pretraining.
    Returns encoded : (total_rows, 17) float32  [16 binary lanes + BPM at col 16]
    """
    npy_path = Path(npy_path)
    arr  = np.load(npy_path)
    meta = json.loads(npy_path.with_suffix('.json').read_text())

    lanes   = arr[:, :16]
    bpm_raw = arr[:, 16].astype(np.float32)
    mask_bpm = bpm_raw != 0
    idx  = np.where(mask_bpm, np.arange(len(bpm_raw)), 0)
    np.maximum.accumulate(idx, out=idx)
    bpm_col = bpm_raw[idx] / 100.0

    encoded_lanes = _encode_lanes_v6(lanes)
    bpm_norm = bpm_col / BPM_SCALE
    encoded  = np.concatenate([encoded_lanes, bpm_norm[:, None]], axis=1)  # (total_rows, 17)

    note_counts_row = np.isin(lanes, list(HEAD_TYPES)).sum(axis=1).astype(np.int32)
    return encoded, note_counts_row, meta


def split_hands_v6(encoded: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a v6-encoded chart into two single-hand arrays in P1 format.

    encoded : (total_rows, 17) float32  — output of encode_chart_pretrain_v6
              cols 0-15: binary lane presence, col 16: BPM normalised

    Returns
    -------
    p1_enc : (total_rows, 9)  [scratch, key1..key7, BPM]
    p2_enc : (total_rows, 9)  [scratch, key1..key7, BPM]  — P2 reordered to P1 format
    """
    bpm = encoded[:, 16:17]
    p1_enc = np.concatenate([encoded[:, P1_LANE_COLS],       bpm], axis=1)
    p2_enc = np.concatenate([encoded[:, P2_TO_P1_LANE_COLS], bpm], axis=1)
    return p1_enc, p2_enc


def window_chart_split(
    encoded: np.ndarray,
    note_counts_row: np.ndarray,
    window_bars: int = WINDOW_BARS,
    stride_bars: int = STRIDE_BARS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Split-hand windowing: each chart window is split into P1 and P2 halves.

    Returns
    -------
    p1_windows      : (T, window_rows, 9)  P1 hand in P1 format
    p2_windows      : (T, window_rows, 9)  P2 hand in P1 format
    win_note_counts : (T,) int32  total note events per window (both hands combined)
    """
    p1_enc, p2_enc = split_hands_v6(encoded)
    wr = window_bars * ROWS_PER_BAR
    sr = stride_bars * ROWS_PER_BAR
    total = encoded.shape[0]

    starts = list(range(0, total - wr + 1, sr))
    if not starts:
        empty = np.zeros((0, wr, IN_CHANNELS_HALF), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.int32)

    p1_windows = np.stack([p1_enc[s:s + wr] for s in starts])
    p2_windows = np.stack([p2_enc[s:s + wr] for s in starts])
    counts     = np.array([int(note_counts_row[s:s + wr].sum()) for s in starts],
                          dtype=np.int32)
    return p1_windows, p2_windows, counts


def window_chart_time_half(
    encoded: np.ndarray,
    note_counts_row: np.ndarray,
    win_secs: float = WIN_SECS,
    stride_secs: float = STRIDE_SECS,
    win_rows: int = WIN_ROWS_V7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-based windowing: windows of win_secs remapped to win_rows at REF_BPM=150.

    Returns
    -------
    p1_windows : (T, win_rows, 9)  lanes 0:8 + BPM at col 8
    p2_windows : (T, win_rows, 9)
    win_note_counts : (T,) int32
    """
    total = encoded.shape[0]
    bpm_arr = np.maximum(encoded[:, 16] * BPM_SCALE, 1.0)
    dt = 1.0 / (bpm_arr * ROWS_PER_BAR / (4.0 * 60.0))
    cum_time = np.concatenate([[0.0], np.cumsum(dt)])

    p1_enc, p2_enc = split_hands_v6(encoded)   # (total, 9): cols 0:8=lanes, 8=BPM

    p1_wins, p2_wins, counts = [], [], []
    t_start = 0.0

    while True:
        start_row = int(np.searchsorted(cum_time, t_start, side='left'))
        if start_row >= total:
            break
        t_end   = t_start + win_secs
        end_row = min(int(np.searchsorted(cum_time, t_end, side='left')), total)

        if cum_time[end_row] - cum_time[start_row] < win_secs * 0.5:
            break

        # Remap source rows to win_rows output rows via physical time
        src_bpm = bpm_arr[start_row:end_row]
        src_dt  = 1.0 / (src_bpm * ROWS_PER_BAR / (4.0 * 60.0))
        elapsed = np.cumsum(src_dt) - src_dt
        out_idx = (elapsed * ROWS_PER_SEC_REF).astype(np.int32)
        valid   = (out_idx >= 0) & (out_idx < win_rows)

        p1_out = np.zeros((win_rows, 8), dtype=np.float32)
        p2_out = np.zeros((win_rows, 8), dtype=np.float32)
        np.maximum.at(p1_out, out_idx[valid], p1_enc[start_row:end_row, :8][valid])
        np.maximum.at(p2_out, out_idx[valid], p2_enc[start_row:end_row, :8][valid])

        bpm_norm = float(src_bpm.mean()) / BPM_SCALE
        bpm_col  = np.full((win_rows, 1), bpm_norm, dtype=np.float32)
        p1_wins.append(np.concatenate([p1_out, bpm_col], axis=1))
        p2_wins.append(np.concatenate([p2_out, bpm_col], axis=1))
        counts.append(int(note_counts_row[start_row:end_row].sum()))

        t_start += stride_secs

    if not p1_wins:
        empty = np.zeros((0, win_rows, IN_CHANNELS_HALF), dtype=np.float32)
        return empty, empty, np.zeros(0, dtype=np.int32)

    return (np.stack(p1_wins), np.stack(p2_wins),
            np.array(counts, dtype=np.int32))


def window_chart_time_half_v10(
    encoded: np.ndarray,
    note_counts_row: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v10 inference windowing: 4s windows, 2s stride, 480-row output."""
    return window_chart_time_half(
        encoded, note_counts_row,
        win_secs=WIN_SECS_V10, stride_secs=STRIDE_SECS_V10, win_rows=WIN_ROWS_V10,
    )


def apply_lane_perm_v3(windows: np.ndarray, perm) -> np.ndarray:
    """
    windows : (..., 17)  last dim = [lane0, ..., lane15, bpm]
    perm    : list of 16 ints — source lane for each output lane position
    """
    return windows[..., list(perm) + [16]]


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
        fat_binary: bool = True,
        encoding_v3: bool = False,
        encoding_v6: bool = False,
    ):
        self.data_root    = Path(data_root)
        self.augment      = augment
        self.window_bars  = window_bars
        self.stride_bars  = stride_bars
        self._encoding_v3 = encoding_v3
        self._encoding_v6 = encoding_v6
        if encoding_v6:
            self._encode    = encode_chart_pretrain_v6
            self._bpm_col   = 16
            self._lane_cols = 16
        elif encoding_v3:
            self._encode    = encode_chart_pretrain_v3
            self._bpm_col   = 16
            self._lane_cols = 16
        else:
            self._encode    = encode_chart_pretrain if fat_binary else encode_chart
            self._bpm_col   = 32
            self._lane_cols = 32

        manifest = pd.read_csv(manifest_csv)
        manifest = manifest[manifest['status'] == 'ok'].reset_index(drop=True)

        self.level_dir = {10: 'dp10_active', 11: 'dp11_active', 12: 'dp12_active'}

        wr = window_bars * ROWS_PER_BAR
        sr = stride_bars * ROWS_PER_BAR

        # Phase 1: collect valid (path_str, npy) pairs from manifest.
        path_rows: list[tuple[str, Path]] = []
        for _, row in manifest.iterrows():
            level_dir = self.data_root / self.level_dir[int(row['level'])]
            npy = level_dir / str(row['file_path'])
            if npy.exists():
                path_rows.append((str(npy), npy))

        unique_paths = sorted({p for p, _ in path_rows})

        # Phase 2: preload all charts into a single contiguous fp16 shared-memory tensor.
        # One tensor = one OS fd; avoids fd exhaustion with forkserver workers.
        # fp16 halves peak RAM; values are in [0,1] so precision is fine.
        print(f"  preloading {len(unique_paths)} charts into shared memory ...", flush=True)
        _enc_list: list[np.ndarray] = []
        _nc_dict:  dict[str, np.ndarray] = {}
        self._offsets: dict[str, int] = {}
        _row = 0
        for path_str in unique_paths:
            enc, note_counts_row, _ = self._encode(path_str)
            _enc_list.append(enc.astype(np.float16))
            _nc_dict[path_str] = note_counts_row
            self._offsets[path_str] = _row
            _row += enc.shape[0]
        _big = np.concatenate(_enc_list, axis=0)
        del _enc_list
        self._encode.cache_clear()  # float32 cache is redundant once _shared is built
        self._shared = torch.from_numpy(_big).share_memory_()
        del _big
        print(f"  done. {self._shared.shape[0]:,} rows, "
              f"{self._shared.nbytes / 1e9:.2f} GB shared", flush=True)

        # Phase 3: build flat index.
        # Each entry stores (npy_path, start_row, density_log, bpm_norm).
        # density_log = log(1 + notes/sec) for this window.
        # bpm_norm = window-mean BPM / BPM_SCALE (for v6 CLS conditioning).
        self.index: list[tuple[Path, int, float, float]] = []
        self.window_weights: list[float] = []
        for path_str, npy in path_rows:
            if path_str not in _nc_dict:
                continue
            nc     = _nc_dict[path_str]
            offset = self._offsets[path_str]
            total_rows = nc.shape[0]
            for start in range(0, total_rows - wr + 1, sr):
                count = int(nc[start:start + wr].sum())
                bpm_mean = (float(self._shared[offset + start: offset + start + wr, self._bpm_col]
                                  .float().mean()) * BPM_SCALE)
                bpm_mean = max(bpm_mean, 1.0)
                dur_sec  = WINDOW_BARS * 4.0 / bpm_mean * 60.0
                density_log = float(np.log1p(count / dur_sec))
                self.index.append((npy, start, density_log, bpm_mean / BPM_SCALE))
                self.window_weights.append(float(count) ** 0.5 + 1.0)

        self._aug_perms = _ALL_AUG_PERMS

    def __len__(self):
        base = len(self.index)
        return base * 8 if self.augment else base

    def __getitem__(self, idx):
        aug_id   = idx % 8 if self.augment else 0
        base_idx = idx // 8 if self.augment else idx

        npy_path, start, density_log, bpm_norm = self.index[base_idx]
        wr = self.window_bars * ROWS_PER_BAR
        _off = self._offsets[str(npy_path)]
        window = self._shared[_off + start: _off + start + wr].float().numpy()  # (768, 17 or 33)

        if aug_id > 0:
            perm = self._aug_perms[aug_id - 1]
            if self._encoding_v3 or self._encoding_v6:
                window = apply_lane_perm_v3(window, perm)
            else:
                window = apply_lane_perm(window, perm)

        if self._encoding_v6:
            return (
                torch.from_numpy(window[:, :self._lane_cols].copy()),  # (768, 16)
                torch.tensor(density_log, dtype=torch.float32),
                torch.tensor(bpm_norm, dtype=torch.float32),
            )
        return (
            torch.from_numpy(window[:, :self._lane_cols].copy()),  # (768, 16) or (768, 32)
            torch.tensor(density_log, dtype=torch.float32),
        )


class PretrainDatasetHalf(Dataset):
    """
    MAE pretraining dataset with single-hand (8-lane) encoding.

    Each chart window contributes two entries — P1 and P2 — both mapped to
    P1 format [scratch, key1..key7].  Augmentation: within-hand mirror
    (keys 1–7 reversed, scratch fixed).  Total multiplier: 2 sides × 2 mirror
    states = 4× base windows.

    Returned tensors per item:
      x        : (window_rows, 8)  float32   binary lane values
      density  : ()                float32   log(1 + hand_notes/sec)
      bpm      : ()                float32   normalised window-mean BPM
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
        manifest = manifest[manifest['status'] == 'ok'].reset_index(drop=True)
        level_dir = {10: 'dp10_active', 11: 'dp11_active', 12: 'dp12_active'}

        wr = window_bars * ROWS_PER_BAR
        sr = stride_bars * ROWS_PER_BAR

        path_rows: list[tuple[str, Path]] = []
        for _, row in manifest.iterrows():
            npy = self.data_root / level_dir[int(row['level'])] / str(row['file_path'])
            if npy.exists():
                path_rows.append((str(npy), npy))

        unique_paths = sorted({p for p, _ in path_rows})

        print(f"  preloading {len(unique_paths)} charts into shared memory (half) ...", flush=True)
        _enc_list: list[np.ndarray] = []
        _nc_dict:  dict[str, np.ndarray] = {}  # per-hand note counts: (total_rows, 2)
        self._offsets: dict[str, int] = {}
        _row = 0
        for path_str in unique_paths:
            enc, note_counts_row, _ = encode_chart_pretrain_v6(path_str)
            _enc_list.append(enc.astype(np.float16))
            # per-hand note counts from the raw binary: sum over P1 lanes, P2 lanes
            head_arr = np.array(sorted(HEAD_TYPES), dtype=np.uint32)
            raw = np.load(path_str)[:, :16]
            p1_nc = np.isin(raw[:, :8],  head_arr).sum(axis=1).astype(np.int32)
            p2_nc = np.isin(raw[:, 8:],  head_arr).sum(axis=1).astype(np.int32)
            _nc_dict[path_str] = np.stack([p1_nc, p2_nc], axis=1)  # (rows, 2)
            self._offsets[path_str] = _row
            _row += enc.shape[0]

        _big = np.concatenate(_enc_list, axis=0)
        del _enc_list
        encode_chart_pretrain_v6.cache_clear()
        self._shared = torch.from_numpy(_big).share_memory_()
        del _big
        print(f"  done. {self._shared.shape[0]:,} rows  "
              f"{self._shared.nbytes / 1e9:.2f} GB shared", flush=True)

        # Index: (npy_path, window_start, side)  side∈{0=P1,1=P2}
        # density_log and bpm_norm computed per hand
        self.index: list[tuple[Path, int, int, float, float]] = []
        self.window_weights: list[float] = []

        for path_str, npy in path_rows:
            if path_str not in _nc_dict:
                continue
            nc2    = _nc_dict[path_str]      # (rows, 2)
            offset = self._offsets[path_str]
            total_rows = nc2.shape[0]

            for start in range(0, total_rows - wr + 1, sr):
                bpm_mean = (float(self._shared[offset + start: offset + start + wr, 16]
                                  .float().mean()) * BPM_SCALE)
                bpm_mean = max(bpm_mean, 1.0)
                bpm_norm = bpm_mean / BPM_SCALE
                dur_sec  = window_bars * 4.0 / bpm_mean * 60.0

                for side in (0, 1):
                    count = int(nc2[start:start + wr, side].sum())
                    density_log = float(np.log1p(count / dur_sec))
                    self.index.append((npy, start, side, density_log, bpm_norm))
                    self.window_weights.append(float(count) ** 0.5 + 1.0)

    def __len__(self):
        base = len(self.index)
        return base * 2 if self.augment else base  # ×2 for mirror

    def __getitem__(self, idx):
        aug_id   = idx % 2 if self.augment else 0  # 0=identity, 1=mirror
        base_idx = idx // 2 if self.augment else idx

        npy_path, start, side, density_log, bpm_norm = self.index[base_idx]
        wr  = self.window_bars * ROWS_PER_BAR
        off = self._offsets[str(npy_path)]
        window = self._shared[off + start: off + start + wr].float().numpy()  # (768, 17)

        cols = P1_LANE_COLS if side == 0 else P2_TO_P1_LANE_COLS
        half = window[:, cols]  # (768, 8)

        if aug_id == 1:
            half = half[:, MIRROR_PERM_HALF]  # flip keys 1–7

        return (
            torch.from_numpy(half.copy()),                      # (768, 8)
            torch.tensor(density_log, dtype=torch.float32),
            torch.tensor(bpm_norm,    dtype=torch.float32),
        )


class PretrainDatasetHalfTime(Dataset):
    """
    MAE pretraining dataset: single-hand (8-lane), BPM-agnostic time-based windows.

    Each window spans exactly WIN_SECS (8 s) of gameplay regardless of BPM.
    Source rows are remapped into WIN_ROWS_V7 (960) output rows at REF_BPM (150).
    High-BPM charts are compressed; low-BPM charts are stretched — all windows share
    the same physical-time scale, eliminating BPM as a confound in pattern similarity.

    Stride = STRIDE_SECS (4 s).  Windows shorter than WIN_SECS * 0.5 at end of chart
    are discarded.  Per-side P1/P2 split and mirror augmentation (×2) are preserved.

    Returned tensors per item:
      x       : (WIN_ROWS_V7, 8) float32   remapped binary lane values
      density : ()               float32   log(1 + hand_notes / WIN_SECS)
      bpm     : ()               float32   normalised window-mean BPM
    """

    def __init__(
        self,
        manifest_csv: str,
        data_root: str,
        augment: bool = True,
        win_secs: float = WIN_SECS,
        stride_secs: float = STRIDE_SECS,
    ):
        self.data_root   = Path(data_root)
        self.augment     = augment
        self.win_secs    = win_secs
        self.stride_secs = stride_secs
        self.win_rows    = int(win_secs * REF_BPM / 60.0 * ROWS_PER_BAR / 4)

        manifest = pd.read_csv(manifest_csv)
        manifest = manifest[manifest['status'] == 'ok'].reset_index(drop=True)
        level_dir = {10: 'dp10_active', 11: 'dp11_active', 12: 'dp12_active'}

        path_rows: list[tuple[str, Path]] = []
        for _, row in manifest.iterrows():
            npy = self.data_root / level_dir[int(row['level'])] / str(row['file_path'])
            if npy.exists():
                path_rows.append((str(npy), npy))

        unique_paths = sorted({p for p, _ in path_rows})

        print(f"  preloading {len(unique_paths)} charts into shared memory (half-time) ...",
              flush=True)
        _enc_list: list[np.ndarray] = []
        _nc_dict:  dict[str, np.ndarray] = {}
        self._offsets: dict[str, int] = {}
        _row = 0
        head_arr = np.array(sorted(HEAD_TYPES), dtype=np.uint32)
        for path_str in unique_paths:
            enc, _, _ = encode_chart_pretrain_v6(path_str)
            _enc_list.append(enc.astype(np.float16))
            raw = np.load(path_str)[:, :16]
            p1_nc = np.isin(raw[:, :8],  head_arr).sum(axis=1).astype(np.int32)
            p2_nc = np.isin(raw[:, 8:],  head_arr).sum(axis=1).astype(np.int32)
            _nc_dict[path_str] = np.stack([p1_nc, p2_nc], axis=1)  # (rows, 2)
            self._offsets[path_str] = _row
            _row += enc.shape[0]

        _big = np.concatenate(_enc_list, axis=0)
        del _enc_list
        encode_chart_pretrain_v6.cache_clear()
        self._shared = torch.from_numpy(_big).share_memory_()
        del _big
        print(f"  done. {self._shared.shape[0]:,} rows  "
              f"{self._shared.nbytes / 1e9:.2f} GB shared", flush=True)

        # Build time-based window index.
        # Each entry: (npy_path, shared_offset, start_row, end_row, side, density_log, bpm_norm)
        self.index: list[tuple[Path, int, int, int, int, float, float]] = []
        self.window_weights: list[float] = []

        for path_str, npy in path_rows:
            if path_str not in _nc_dict:
                continue
            nc2    = _nc_dict[path_str]       # (rows, 2) per-hand note counts
            offset = self._offsets[path_str]
            total_rows = nc2.shape[0]

            # BPM per row from shared tensor (col 16 = bpm / BPM_SCALE, float16 → float32)
            bpm_arr = np.maximum(
                self._shared[offset:offset + total_rows, 16].float().numpy() * BPM_SCALE,
                1.0,
            )
            dt = 1.0 / (bpm_arr * ROWS_PER_BAR / (4.0 * 60.0))  # seconds per row
            cum_time = np.concatenate([[0.0], np.cumsum(dt)])      # (rows+1,) float64

            t_start = 0.0
            while True:
                start_row = int(np.searchsorted(cum_time, t_start, side='left'))
                if start_row >= total_rows:
                    break
                t_end    = t_start + self.win_secs
                end_row  = min(int(np.searchsorted(cum_time, t_end, side='left')), total_rows)

                if cum_time[end_row] - cum_time[start_row] < self.win_secs * 0.5:
                    break

                bpm_norm = float(bpm_arr[start_row:end_row].mean()) / BPM_SCALE

                for side in (0, 1):
                    count = int(nc2[start_row:end_row, side].sum())
                    density_log = float(np.log1p(count / self.win_secs))
                    self.index.append(
                        (npy, offset, start_row, end_row, side, density_log, bpm_norm)
                    )
                    self.window_weights.append(float(count) ** 0.5 + 1.0)

                t_start += self.stride_secs

    def __len__(self):
        base = len(self.index)
        return base * 2 if self.augment else base  # ×2 for mirror

    def __getitem__(self, idx):
        aug_id   = idx % 2 if self.augment else 0
        base_idx = idx // 2 if self.augment else idx

        npy_path, offset, start_row, end_row, side, density_log, bpm_norm = self.index[base_idx]

        src = self._shared[offset + start_row : offset + end_row].float().numpy()  # (L, 17)

        # Vectorised remap: elapsed time at start of each row → output row index
        bpm_arr  = np.maximum(src[:, 16] * BPM_SCALE, 1.0)
        dt       = 1.0 / (bpm_arr * ROWS_PER_BAR / (4.0 * 60.0))
        elapsed  = np.cumsum(dt) - dt        # time at row start, 0-based from window start
        out_idx  = (elapsed * ROWS_PER_SEC_REF).astype(np.int32)
        valid    = (out_idx >= 0) & (out_idx < self.win_rows)

        cols = P1_LANE_COLS if side == 0 else P2_TO_P1_LANE_COLS
        half = src[:, cols]                  # (L, 8)

        out = np.zeros((self.win_rows, 8), dtype=np.float32)
        np.maximum.at(out, out_idx[valid], half[valid])

        if aug_id == 1:
            out = out[:, MIRROR_PERM_HALF]

        return (
            torch.from_numpy(out.copy()),
            torch.tensor(density_log, dtype=torch.float32),
            torch.tensor(bpm_norm,    dtype=torch.float32),
        )


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

        self._aug_perms = [None] + _ALL_AUG_PERMS

    def __len__(self):
        return len(self.entries) * (8 if self.augment else 1)

    def __getitem__(self, idx):
        aug_id   = idx % 8 if self.augment else 0
        base_idx = idx // 8 if self.augment else idx

        entry = self.entries[base_idx]
        encoded, note_counts_row, _ = encode_chart(str(entry['npy']))
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
