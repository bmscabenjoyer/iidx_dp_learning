"""
Chord distribution analysis across all lv12 charts — single-hand view.

Each hand (P1: lanes 1–7, P2: lanes 8–14) is read independently per row.
Scratches (lanes 0, 15) are excluded.
P2 key lanes (8–14) are mapped to key positions 1–7 (lane - 7) so both
hands share the same position space.

A chord is the set of keys one hand is pressing simultaneously.
Each row contributes up to two chords (one per hand) if both are active.

Lane layout (input):
  0        = P1 scratch  (excluded)
  1–7      = P1 keys 1–7 → positions 1–7
  8–14     = P2 keys 1–7 → positions 1–7  (lane - 7)
  15       = P2 scratch  (excluded)
"""

import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter

DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
MANIFEST  = DATA_ROOT / 'labeled_manifest.csv'
DP12_DIR  = DATA_ROOT / 'dp12_active'

ACTION_VALUES = {1, 2, 4, 6, 8}  # tap, CN head, HCN head, BSS head, MSS head
P1_LANES      = range(1, 8)       # lanes 1–7
P2_LANES      = range(8, 15)      # lanes 8–14

def chord_label(positions):
    """Human-readable label from sorted key positions."""
    return '+'.join(str(p) for p in positions)


def main():
    manifest = pd.read_csv(MANIFEST)
    lv12 = manifest[
        (manifest['level'] == 12) &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)
    print(f'lv12 charts: {len(lv12)}')

    chord_counter = Counter()
    total_charts  = 0
    skipped       = 0

    for _, row in lv12.iterrows():
        npy = DP12_DIR / row['file_path']
        if not npy.exists():
            skipped += 1
            continue

        chart = np.load(npy)      # (rows, 17)
        lanes = chart[:, :16]     # drop BPM column

        for r in range(len(lanes)):
            p1 = tuple(l       for l in P1_LANES if int(lanes[r, l]) in ACTION_VALUES)
            p2 = tuple(l - 7   for l in P2_LANES if int(lanes[r, l]) in ACTION_VALUES)
            if p1:
                chord_counter[p1] += 1
            if p2:
                chord_counter[p2] += 1

        total_charts += 1

    print(f'charts loaded: {total_charts}  (skipped: {skipped})')

    total_events = sum(chord_counter.values())
    print(f'total single-hand chord events: {total_events:,}')
    print(f'unique chord types: {len(chord_counter):,}')

    # ── distribution by chord size ────────────────────────────────────────────
    size_counter = Counter()
    for chord, cnt in chord_counter.items():
        size_counter[len(chord)] += cnt

    print(f'\n── chord size distribution ──────────────────────────')
    print(f'  {"notes":>5}  {"count":>10}  {"pct":>7}')
    for size in sorted(size_counter):
        cnt = size_counter[size]
        print(f'  {size:>5}  {cnt:>10,}  {100*cnt/total_events:>6.2f}%')

    # ── symmetry breakdown ────────────────────────────────────────────────────
    # A chord is symmetric if it contains exactly 2 of each position (one per hand)
    # A chord is one-sided if all positions are from one side only (need raw lane data)
    # Approximate: symmetric chords have even multiplicity for all positions
    print(f'\n── top 60 chords (key positions only, P2 mapped to P1) ──')
    print(f'  {"chord":<30}  {"notes":>5}  {"count":>9}  {"pct":>6}')
    for chord, cnt in chord_counter.most_common(60):
        label = chord_label(chord)
        print(f'  {label:<30}  {len(chord):>5}  {cnt:>9,}  {100*cnt/total_events:>5.2f}%')

    # ── coverage ─────────────────────────────────────────────────────────────
    print(f'\n── coverage ─────────────────────────────────────────')
    print(f'   appearing ≥100 times: {sum(1 for c in chord_counter.values() if c >= 100):,}')
    print(f'   appearing  ≥10 times: {sum(1 for c in chord_counter.values() if c >= 10):,}')
    print(f'   appearing   once    : {sum(1 for c in chord_counter.values() if c == 1):,}')

    cumulative = 0
    print(f'\n── cumulative coverage by top-N chord types ─────────')
    print(f'  {"top-N":>6}  {"cumulative %":>13}')
    for n, (chord, cnt) in enumerate(chord_counter.most_common(), 1):
        cumulative += cnt
        if n in {10, 20, 50, 100, 200, 500}:
            print(f'  {n:>6}  {100*cumulative/total_events:>12.2f}%')
        if n > 500:
            break


if __name__ == '__main__':
    main()
