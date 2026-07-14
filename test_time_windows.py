"""
Test time-based (BPM-agnostic) windowing vs current bar-based windowing.
Compare on Mare Nectaris [DP ANOTHER] (BPM=256, ec=10.6).

New windowing:
  - Each window = 8 seconds of gameplay (= 5 bars at 150 BPM)
  - Source rows accumulated until 8s elapsed (varies by BPM)
  - Remapped into fixed 960-row output: output_row = time_elapsed * 120
    (150 BPM × 192 rows/bar ÷ (4 beats × 60 sec/min) = 120 rows/sec)
  - Stride = 4 seconds
  - No BPM conditioning needed
"""

import numpy as np
from pathlib import Path

DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
NPY_PATH  = DATA_ROOT / 'dp12_active/charts/Mare Nectaris_24_marenect_DAC00.npy'

ROWS_PER_BAR  = 192
# Old config
OLD_WIN_BARS   = 4
OLD_WIN_ROWS   = OLD_WIN_BARS * ROWS_PER_BAR   # 768
OLD_STRIDE_BARS = 2
OLD_STRIDE_ROWS = OLD_STRIDE_BARS * ROWS_PER_BAR  # 384
# New config
REF_BPM        = 150.0
WIN_SECS       = 8.0
STRIDE_SECS    = 4.0
NEW_WIN_ROWS   = int(WIN_SECS * REF_BPM / 60 * ROWS_PER_BAR / 4)   # 960
ROWS_PER_SEC   = REF_BPM * ROWS_PER_BAR / (4 * 60)                  # 120


def load_chart(npy_path):
    arr     = np.load(npy_path)          # (total_rows, 17)
    lanes   = arr[:, :16].astype(np.uint8)
    bpm_raw = arr[:, 16].astype(np.float32)

    # fill-forward BPM
    bpm = np.zeros(len(arr), dtype=np.float32)
    cur = 150.0
    for i in range(len(arr)):
        if bpm_raw[i] > 0:
            cur = bpm_raw[i] / 100.0
        bpm[i] = cur

    return lanes, bpm


def old_windows(lanes, bpm):
    """Bar-based windows: fixed 768 rows, stride 384."""
    total = len(lanes)
    starts = list(range(0, total - OLD_WIN_ROWS + 1, OLD_STRIDE_ROWS))
    wins, note_counts, avg_bpms = [], [], []
    for s in starts:
        win = lanes[s:s + OLD_WIN_ROWS]
        wins.append(win)
        note_counts.append(int((win > 0).sum()))
        avg_bpms.append(float(bpm[s:s + OLD_WIN_ROWS].mean()))
    return wins, note_counts, avg_bpms


def new_windows(lanes, bpm):
    """
    Time-based windows: always 8 seconds.
    Walk rows accumulating time; when 8s elapsed, end window.
    Remap source rows into 960-row output via: out_row = int(elapsed_time * ROWS_PER_SEC)
    Stride = 4 seconds.
    """
    total = len(lanes)

    # pre-compute cumulative time (seconds) at each row boundary
    dt = 1.0 / (bpm * ROWS_PER_BAR / (4 * 60))   # seconds per row
    cum_time = np.concatenate([[0.0], np.cumsum(dt)])  # (total+1,)

    wins, note_counts, avg_bpms, src_rows_counts = [], [], [], []

    stride_sec = STRIDE_SECS
    t_start = 0.0

    while True:
        # find row at t_start
        start_row = int(np.searchsorted(cum_time, t_start, side='left'))
        if start_row >= total:
            break
        t_end = t_start + WIN_SECS
        end_row = int(np.searchsorted(cum_time, t_end, side='left'))
        end_row = min(end_row, total)

        # check we have enough content
        actual_secs = cum_time[end_row] - cum_time[start_row]
        if actual_secs < WIN_SECS * 0.5:  # less than half window at end of chart
            break

        # remap source rows into 960-row output
        out = np.zeros((NEW_WIN_ROWS, 16), dtype=np.uint8)
        t0 = cum_time[start_row]
        for r in range(start_row, end_row):
            elapsed = cum_time[r] - t0
            out_row = int(elapsed * ROWS_PER_SEC)
            if 0 <= out_row < NEW_WIN_ROWS:
                # OR: if multiple source rows map to same output row, keep any active note
                out[out_row] = np.maximum(out[out_row], lanes[r])

        wins.append(out)
        note_counts.append(int((out > 0).sum()))
        avg_bpms.append(float(bpm[start_row:end_row].mean()))
        src_rows_counts.append(end_row - start_row)

        t_start += stride_sec

    return wins, note_counts, avg_bpms, src_rows_counts


def piano_roll_compact(win, n_rows=32):
    """Show compressed piano roll: group rows into n_rows buckets."""
    total = len(win)
    bucket = total // n_rows
    lines = []
    for i in range(n_rows):
        chunk = win[i*bucket:(i+1)*bucket]
        active = (chunk > 0).any(axis=0)  # (16,)
        cells = []
        for lane in range(16):
            if active[lane]:
                if lane == 0 or lane == 15:
                    cells.append('S')
                else:
                    k = lane if lane < 8 else lane - 7
                    cells.append(str(k))
            else:
                cells.append('.')
        sep = '|' if i % 8 == 0 else ' '
        lines.append(sep + ''.join(cells[:8]) + '│' + ''.join(cells[8:]))
    return lines


# ── Run comparison ────────────────────────────────────────────────────────────

lanes, bpm = load_chart(NPY_PATH)
total_rows  = len(lanes)
total_secs  = float(sum(1.0 / (bpm * ROWS_PER_BAR / (4 * 60))))

print(f'Mare Nectaris [DP ANOTHER]')
print(f'  total rows : {total_rows}')
print(f'  BPM        : {bpm[bpm > 0].min():.0f} – {bpm.max():.0f}  '
      f'(mean {bpm.mean():.1f})')
print(f'  duration   : {total_secs:.1f}s  '
      f'({total_rows/ROWS_PER_BAR:.1f} bars)')

old_wins, old_nc, old_bpms = old_windows(lanes, bpm)
new_wins, new_nc, new_bpms, new_src = new_windows(lanes, bpm)

print(f'\nOld windowing (4-bar, stride 2-bar):')
print(f'  {len(old_wins)} windows  ×  {OLD_WIN_ROWS} rows each')
print(f'\nNew windowing (8-sec, stride 4-sec):')
print(f'  {len(new_wins)} windows  ×  {NEW_WIN_ROWS} rows each  '
      f'(src rows: {min(new_src)}–{max(new_src)} per window)')

# ── Side-by-side comparison of first 6 windows ───────────────────────────────

print(f'\n{"═"*90}')
print(f'  Side-by-side: old (left) vs new (right) for first 6 windows')
print(f'  Each row = 1/{32} of window. P1: S 1234567 │ P2: 1234567 S')
print(f'{"═"*90}')

n_show = min(6, len(old_wins), len(new_wins))
for i in range(n_show):
    old_bars = OLD_WIN_ROWS / ROWS_PER_BAR
    new_secs = WIN_SECS

    old_header = (f'  OLD win {i+1}  bars {i*OLD_STRIDE_BARS+1}–'
                  f'{i*OLD_STRIDE_BARS+OLD_WIN_BARS}  '
                  f'bpm={old_bpms[i]:.0f}  notes={old_nc[i]}')
    new_header = (f'  NEW win {i+1}  t={i*STRIDE_SECS:.0f}–'
                  f'{i*STRIDE_SECS+WIN_SECS:.0f}s  '
                  f'bpm={new_bpms[i]:.0f}  notes={new_nc[i]}  '
                  f'src={new_src[i]}rows')

    print(f'\n  {"─"*42}    {"─"*42}')
    print(f'{old_header:<46}  {new_header}')
    print(f'  {"─"*42}    {"─"*42}')

    old_lines = piano_roll_compact(old_wins[i])
    new_lines = piano_roll_compact(new_wins[i])
    for a, b in zip(old_lines, new_lines):
        print(f'  {a:<44}  {b}')

# ── Note count comparison across all windows ─────────────────────────────────

print(f'\n{"─"*70}')
print(f'  Window-by-window note counts')
print(f'  {"old win":>8}  {"notes":>6}  {"bpm":>6}    {"new win":>8}  {"notes":>6}  '
      f'{"bpm":>6}  {"src rows":>9}')
print(f'  {"─"*8}  {"─"*6}  {"─"*6}    {"─"*8}  {"─"*6}  {"─"*6}  {"─"*9}')
for i in range(max(len(old_wins), len(new_wins))):
    o = (f'{i+1:>8}  {old_nc[i]:>6}  {old_bpms[i]:>6.0f}'
         if i < len(old_wins) else f'{"":>8}  {"":>6}  {"":>6}')
    n = (f'{i+1:>8}  {new_nc[i]:>6}  {new_bpms[i]:>6.0f}  {new_src[i]:>9}'
         if i < len(new_wins) else '')
    print(f'  {o}    {n}')

# ── Drift check: where do the first 5 notes of the chart land? ───────────────

print(f'\n{"─"*70}')
print(f'  Drift check: first 10 note events — row position in old vs new win 1')

note_rows = np.where((lanes > 0).any(axis=1))[0][:10]
old_win0_start = 0
# new win 0 start row
dt = 1.0 / (bpm * ROWS_PER_BAR / (4 * 60))
cum_time = np.concatenate([[0.0], np.cumsum(dt)])

print(f'  {"note":>5}  {"src row":>8}  {"old out_row":>11}  {"old pos%":>9}  '
      f'{"new out_row":>11}  {"new pos%":>9}  {"match":>6}')
print(f'  {"─"*5}  {"─"*8}  {"─"*11}  {"─"*9}  {"─"*11}  {"─"*9}  {"─"*6}')

new_win0_t0 = cum_time[0]
for k, r in enumerate(note_rows):
    old_out = r - old_win0_start
    elapsed = cum_time[r] - new_win0_t0
    new_out = int(elapsed * ROWS_PER_SEC)
    old_pct = 100.0 * old_out / OLD_WIN_ROWS
    new_pct = 100.0 * new_out / NEW_WIN_ROWS
    match   = '✓' if abs(old_pct - new_pct) < 2.0 else f'Δ{new_pct-old_pct:+.1f}%'
    print(f'  {k+1:>5}  {r:>8}  {old_out:>11}  {old_pct:>8.1f}%  '
          f'{new_out:>11}  {new_pct:>8.1f}%  {match:>6}')
