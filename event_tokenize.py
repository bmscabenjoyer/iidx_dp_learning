"""
Convert an IIDX DP chart (.npy) to a per-timestep event token sequence.

One token = one unique active timestep (all lanes firing simultaneously).

Stored fields per token
-----------------------
lane_mask  : uint16  bitmask of active lanes (bit i set ⟺ lane i has a head note)
note_type  : uint8   dominant note type: 0=tap  1=cn  2=hcn  3=bss/mss
times_sec  : float32 absolute physical time of this timestep from chart start

Delta bins are NOT stored here — they are computed per-window inside the
dataset to avoid cross-window leakage on the first token of each window.

Delta quantisation (for reference / standalone use)
---------------------------------------------------
Note speed, linear-scale, 32 bins:
  bin 0          : start sentinel (first event in window, delta = 0)
  bins 1–31      : linear note speed 0 → MAX_SPEED_NPS (25 notes/sec)
  150 BPM 16th   → bin 13  (100 ms, 10 notes/sec)
  200 BPM 16th   → bin 17  (75 ms,  13.3 notes/sec)
  100 BPM 16th   → bin  9  (150 ms, 6.7 notes/sec)
  150 vs 160 BPM 16th → same bin 13 (similar BPM, similar speed)
  100 vs 200 BPM 16th → bins 9 vs 17 (Δ=8, clearly different)

Usage
-----
    python event_tokenize.py --chart-dir ~/projects/iidx_data/dp12_active/charts \\
                             --output-dir cache/events/dp12

    # or all three levels
    for L in 10 11 12; do
        python event_tokenize.py \\
            --chart-dir ~/projects/iidx_data/dp${L}_active/charts \\
            --output-dir cache/events/dp${L}
    done
"""

import math
import argparse
import numpy as np
from pathlib import Path

# ── constants ─────────────────────────────────────────────────────────────────

ROWS_PER_BEAT = 48
BPM_COL       = 16
HEAD_VALUES   = frozenset({1, 2, 4, 6, 8})

DELTA_BINS    = 32
MAX_SPEED_NPS = 25.0    # notes/sec ceiling (≡ 40 ms minimum gap → bin 31)

# note_type priority (higher overrides lower within a timestep)
_TYPE_PRIORITY = {1: 1, 2: 2, 4: 3, 6: 1, 8: 1}   # tap/bss/mss=1, cn=2, hcn=3
_TYPE_ID       = {1: 0, 2: 1, 4: 2, 6: 3, 8: 3}    # value → type id


# ── helpers ───────────────────────────────────────────────────────────────────

def _fill_bpm(bpm_raw: np.ndarray, default: float = 150.0) -> np.ndarray:
    """Fill-forward BPM column (stored as BPM×100, 0 = no change)."""
    out = np.empty(len(bpm_raw), dtype=np.float64)
    cur = default
    for i, v in enumerate(bpm_raw):
        if v > 0:
            cur = float(v) / 100.0
        out[i] = cur
    return out


def _row_times(bpm: np.ndarray) -> np.ndarray:
    """Physical time (seconds) at the start of each row."""
    dt = 60.0 / (bpm * ROWS_PER_BEAT)
    cum = np.empty(len(dt), dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(dt[:-1], out=cum[1:])
    return cum


def quantize_delta(delta_sec: float) -> int:
    """Single-value note-speed delta. bin 0 = start sentinel (delta ≤ 0)."""
    if delta_sec <= 0.0:
        return 0
    speed = min(1.0 / delta_sec, MAX_SPEED_NPS)
    return min(int(speed / MAX_SPEED_NPS * 30) + 1, 31)


def quantize_delta_vec(deltas_sec: np.ndarray) -> np.ndarray:
    """Vectorised version; returns uint8 array of same length."""
    out = np.zeros(len(deltas_sec), dtype=np.uint8)
    pos = deltas_sec > 0.0
    if not pos.any():
        return out
    speed = np.minimum(1.0 / deltas_sec[pos], MAX_SPEED_NPS)
    bins  = np.clip((speed / MAX_SPEED_NPS * 30).astype(np.int32) + 1, 1, 31)
    out[pos] = bins.astype(np.uint8)
    return out


# ── core tokeniser ────────────────────────────────────────────────────────────

def chart_to_tokens(npy_path: str | Path) -> dict:
    """
    Tokenise a single chart.

    Returns
    -------
    dict with keys:
      lane_masks : (N,) uint16  — bitmask of active lanes
      note_types : (N,) uint8   — dominant note type (0 tap 1 cn 2 hcn 3 bss/mss)
      times_sec  : (N,) float32 — physical time of each timestep in seconds
    where N = number of unique active timesteps in the chart.
    """
    arr = np.load(str(npy_path))                              # (rows, 17)
    bpm = _fill_bpm(arr[:, BPM_COL].astype(np.float64))
    times = _row_times(bpm)                                   # (rows,) seconds

    # Active rows: any lane in 0-15 has a head note
    head_arr = np.isin(arr[:, :16], list(HEAD_VALUES))        # (rows, 16) bool
    active   = np.where(head_arr.any(axis=1))[0]             # active row indices

    if len(active) == 0:
        return {
            'lane_masks': np.empty(0, dtype=np.uint16),
            'note_types': np.empty(0, dtype=np.uint8),
            'times_sec':  np.empty(0, dtype=np.float32),
        }

    vals = arr[active, :16]                                   # (N, 16) note values

    # Lane bitmasks: bit i set ⟺ lane i has a head note
    is_head = np.isin(vals, list(HEAD_VALUES))                # (N, 16) bool
    powers  = (1 << np.arange(16, dtype=np.int32))
    masks   = (is_head.astype(np.int32) * powers).sum(axis=1).astype(np.uint16)

    # Dominant note type per timestep
    # Priority: hcn(3) > cn(2) > tap/bss/mss(1); ties broken by type_id
    priority = np.vectorize(_TYPE_PRIORITY.get)(vals, 0) * is_head  # (N, 16)
    type_id  = np.vectorize(_TYPE_ID.get)(vals, 0) * is_head        # (N, 16)
    best_col = priority.argmax(axis=1)                               # (N,)
    ntypes   = type_id[np.arange(len(active)), best_col].astype(np.uint8)

    return {
        'lane_masks': masks,
        'note_types': ntypes,
        'times_sec':  times[active].astype(np.float32),
    }


# ── cache builder ─────────────────────────────────────────────────────────────

def build_cache(
    chart_dir:  str,
    output_dir: str,
    glob:       str  = '*.npy',
    overwrite:  bool = False,
) -> None:
    """Tokenise all charts in chart_dir, save each as <stem>.npz in output_dir."""
    src = Path(chart_dir).expanduser()
    dst = Path(output_dir).expanduser()
    dst.mkdir(parents=True, exist_ok=True)

    paths   = sorted(src.glob(glob))
    n_done  = n_skip = n_err = 0

    print(f'Tokenising {len(paths)} charts  →  {dst}')
    for p in paths:
        out = dst / (p.stem + '.npz')
        if out.exists() and not overwrite:
            n_skip += 1
            continue
        try:
            toks = chart_to_tokens(p)
            np.savez_compressed(out, **toks)
            n_done += 1
        except Exception as e:
            print(f'  ERR {p.name}: {e}')
            n_err += 1

    print(f'  done={n_done}  skipped={n_skip}  errors={n_err}')


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--chart-dir',  required=True, help='directory of .npy chart files')
    ap.add_argument('--output-dir', required=True, help='directory to write .npz caches')
    ap.add_argument('--overwrite',  action='store_true', help='re-tokenise existing caches')
    args = ap.parse_args()
    build_cache(args.chart_dir, args.output_dir, overwrite=args.overwrite)
