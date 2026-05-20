import json
import functools
import numpy as np
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset

ZASA_MEAN = 11.174  # mean over unified lv10+lv11 zasa + lv12 ereter rating_stat
ZASA_STD  = 0.926   # std over unified distribution

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
PATCH_ROWS_V4            = 12  # sub-beat (16th-note) resolution
NUM_PATCHES_V4           = WINDOW_ROWS // PATCH_ROWS_V4  # 64

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
    Single-channel encoding: 0.0=empty, 1.0=tap (fat-binary smear), 0.5=hold body.
    Scratch lanes (0,15) use narrower smear; tap smear overrides hold body in overlap.
    Returns: (total_rows, 16) float32
    """
    total_rows = lanes.shape[0]
    out = np.zeros((total_rows, 16), dtype=np.float32)

    head_arr = np.array(sorted(HEAD_TYPES), dtype=lanes.dtype)
    head_mask = np.isin(lanes, head_arr)  # (total_rows, 16) bool

    body_arr = np.array(sorted(BODY_SUSTAIN.keys()), dtype=lanes.dtype)
    out[np.isin(lanes, body_arr)] = 0.5   # step 1: hold bodies → 0.5

    key_kern = np.ones(2 * PRETRAIN_KEY_HALF_WIDTH + 1, dtype=np.float32)
    sc_kern  = np.ones(2 * PRETRAIN_SCRATCH_HALF_WIDTH + 1, dtype=np.float32)
    for lane in range(16):
        kern    = sc_kern if lane in SCRATCH_LANES else key_kern
        smeared = np.convolve(head_mask[:, lane].astype(np.float32), kern, mode='same') > 0
        out[:, lane] = np.where(smeared, 1.0, out[:, lane])  # step 2: tap smear → 1.0

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
        zasa_csv: str | None = None,
        encoding_v3: bool = False,
    ):
        self.data_root    = Path(data_root)
        self.augment      = augment
        self.window_bars  = window_bars
        self.stride_bars  = stride_bars
        self._encoding_v3 = encoding_v3
        if encoding_v3:
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

        # Build unified difficulty lookup: absolute npy path → normalised rating.
        # Sources: zasa_csv for lv10/11, manifest rating_stat for lv12.
        # NaN for any chart without a rating (unmatched lv10/11, lv12 missing ereter).
        self._zasa: dict[str, float] = {}
        if zasa_csv is not None:
            zdf = pd.read_csv(zasa_csv)
            for _, row in zdf.iterrows():
                ldir = self.level_dir[int(row['level'])]
                key  = str(Path(data_root) / ldir / str(row['file_path']))
                self._zasa[key] = (float(row['zasa_rating']) - ZASA_MEAN) / ZASA_STD
            print(f"  loaded {len(self._zasa)} zasa ratings (lv10/11)", flush=True)

        # lv12: use ereter rating_stat from manifest (same difficulty scale as zasa)
        lv12_rows = manifest[(manifest['level'] == 12) & manifest['rating_stat'].notna()]
        n_lv12 = 0
        for _, row in lv12_rows.iterrows():
            key = str(Path(data_root) / self.level_dir[12] / str(row['file_path']))
            self._zasa[key] = (float(row['rating_stat']) - ZASA_MEAN) / ZASA_STD
            n_lv12 += 1
        print(f"  loaded {n_lv12} ereter stat ratings (lv12)", flush=True)

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
        # Each entry stores (npy_path, start_row, density_log, zasa_norm).
        # density_log = log(1 + notes/sec) for this window — computed from
        # BPM stored at col 32 of _shared (BPM/BPM_SCALE, fp16).
        # zasa_norm   = (zasa_rating - ZASA_MEAN) / ZASA_STD, or NaN.
        self.index: list[tuple[Path, int, float, float]] = []
        self.window_weights: list[float] = []
        _nan = float('nan')
        for path_str, npy in path_rows:
            if path_str not in _nc_dict:
                continue
            nc     = _nc_dict[path_str]
            offset = self._offsets[path_str]
            total_rows = nc.shape[0]
            zasa_norm  = self._zasa.get(path_str, _nan)
            for start in range(0, total_rows - wr + 1, sr):
                count = int(nc[start:start + wr].sum())
                bpm_mean = (float(self._shared[offset + start: offset + start + wr, self._bpm_col]
                                  .float().mean()) * BPM_SCALE)
                bpm_mean = max(bpm_mean, 1.0)
                dur_sec  = WINDOW_BARS * 4.0 / bpm_mean * 60.0
                density_log = float(np.log1p(count / dur_sec))
                self.index.append((npy, start, density_log, zasa_norm))
                self.window_weights.append(float(count) ** 0.5 + 1.0)

        self._aug_perms = _ALL_AUG_PERMS

    def __len__(self):
        base = len(self.index)
        return base * 8 if self.augment else base

    def __getitem__(self, idx):
        aug_id   = idx % 8 if self.augment else 0
        base_idx = idx // 8 if self.augment else idx

        npy_path, start, density_log, zasa_norm = self.index[base_idx]
        wr = self.window_bars * ROWS_PER_BAR
        _off = self._offsets[str(npy_path)]
        window = self._shared[_off + start: _off + start + wr].float().numpy()  # (768, 33)

        if aug_id > 0:
            perm = self._aug_perms[aug_id - 1]
            window = (apply_lane_perm_v3 if self._encoding_v3 else apply_lane_perm)(window, perm)

        return (
            torch.from_numpy(window[:, :self._lane_cols].copy()),  # (768, 16) or (768, 32)
            torch.tensor(density_log, dtype=torch.float32),    # scalar
            torch.tensor(zasa_norm,   dtype=torch.float32),    # scalar (NaN for lv12)
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
