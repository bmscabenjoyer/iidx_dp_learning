"""
Show exact note positions for quasar windows 27-34 and their top match,
rendered as a compressed piano roll (1/4-beat resolution).

Lanes:  P1: S 1 2 3 4 5 6 7 | P2: 1 2 3 4 5 6 7 S
Each row = 1/4 beat = 12 chart rows.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path

from dataset import (
    encode_chart_pretrain_v6, window_chart_split,
    WINDOW_BARS, STRIDE_BARS,
    PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
    NUM_PATCHES_V4, PATCH_ROWS_V4,
)
from mae import MAEModel

MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
CKPT      = Path('checkpoints/pretrain_half/encoder_best.pt')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 256
ROWS_PER_WIN = 768   # 4 bars × 192 rows
SUBDIV       = 12    # rows per display tick (= 1/4 beat)
TICKS        = ROWS_PER_WIN // SUBDIV   # 64 ticks per window

# windows of interest (1-based)
WIN_RANGE = range(27, 35)


def load_encoder():
    model = MAEModel(
        in_channels=PRETRAIN_IN_CHANNELS_HALF,
        num_patches=NUM_PATCHES_V4, patch_rows=PATCH_ROWS_V4,
        encoder_dim=EMBED_DIM, decoder_dim=128, num_heads=8,
        use_lane_type_embed=True, use_bpm_cond=True,
        lane_types=LANE_TYPES_HALF,
    )
    model.encoder.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    return model.encoder.eval().to(DEVICE)


def raw_windows(npy_path):
    """Return (p1_wins_raw, p2_wins_raw) as (T, 768, 16) note arrays (0-9)."""
    arr = np.load(npy_path)          # (total_rows, 17)
    notes = arr[:, :16].astype(np.uint8)
    bpm_col = arr[:, 16]

    # fill-forward bpm
    bpm = np.zeros(len(arr), dtype=np.float32)
    cur = 150.0
    for i, v in enumerate(bpm_col):
        if v > 0: cur = v / 100.0
        bpm[i] = cur

    enc, nc, _ = encode_chart_pretrain_v6(str(npy_path))
    _, _, win_nc = window_chart_split(enc, nc, WINDOW_BARS, STRIDE_BARS)
    T_enc = len(win_nc)

    stride = int(STRIDE_BARS * 192)
    wins_raw = []
    for t in range(T_enc):
        start = t * stride
        end   = start + ROWS_PER_WIN
        if end > len(notes):
            pad = np.zeros((end - len(notes), 16), dtype=np.uint8)
            chunk = np.concatenate([notes[start:], pad])
        else:
            chunk = notes[start:end]
        wins_raw.append(chunk)
    return wins_raw   # list of (768, 16) arrays


def piano_roll(win_raw, title, win_idx, total, pos_pct, ec=None):
    """
    Render one window as a compressed piano roll string.
    win_raw: (768, 16) uint8 note array
    Returns list of lines.
    """
    # header
    ec_str = f'  ec={ec:.1f}' if ec is not None else ''
    lines = [
        f'  {"─"*70}',
        f'  {title}   win {win_idx}/{total}  ({pos_pct}){ec_str}',
        f'  P1: S  1  2  3  4  5  6  7 │ P2: 1  2  3  4  5  6  7  S',
        f'  {"─"*70}',
    ]

    # beat markers: every 4 ticks = 1 beat (4 beats/bar × 4 bars = 16 beats)
    bar = 0
    for tick in range(TICKS):
        chunk = win_raw[tick*SUBDIV:(tick+1)*SUBDIV]   # (12, 16)
        active = (chunk > 0).any(axis=0)               # (16,) bool

        cells = []
        for lane in range(16):
            if active[lane]:
                if lane == 0 or lane == 15:
                    cells.append(' S ')
                else:
                    k = lane if lane < 8 else lane - 7
                    cells.append(f' {k} ')
            else:
                cells.append(' . ')

        beat_in_bar = (tick // 4) % 4
        sub         = tick % 4

        # beat marker prefix
        if sub == 0:
            beat_num = tick // 4
            bar_num  = beat_num // 4 + 1
            prefix   = f'{bar_num}.{beat_in_bar+1}'
        else:
            prefix   = '    '

        row = ''.join(cells[:8]) + '│' + ''.join(cells[8:])
        lines.append(f'  {prefix:5s} {row}')

    return lines


# ── Load quasar ───────────────────────────────────────────────────────────────

print(f'device: {DEVICE}', flush=True)
encoder  = load_encoder()
manifest = pd.read_csv(MANIFEST)

lv12 = manifest[
    (manifest['level'] == 12) &
    manifest['rating_ec'].notna() &
    (manifest['status'] == 'ok')
].reset_index(drop=True)

# find quasar
q_row = lv12[lv12['title'].str.lower() == 'quasar'].iloc[0]
q_npy = DATA_ROOT / 'dp12_active' / str(q_row['file_path'])
q_wins_raw = raw_windows(q_npy)
q_total    = len(q_wins_raw)

print(f'quasar: {q_total} windows', flush=True)

# embed quasar windows and all other lv12 windows for NN search
enc_arr, nc_arr, _ = encode_chart_pretrain_v6(str(q_npy))
p1_wins, p2_wins, _ = window_chart_split(enc_arr, nc_arr, WINDOW_BARS, STRIDE_BARS)

@torch.no_grad()
def embed_batch(wins):
    x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
    bpm = torch.from_numpy(wins[:, :, 8].mean(1)).float().to(DEVICE)
    return encoder.embed_segment(x, bpm=bpm).cpu().numpy()

q_embs = embed_batch(p1_wins)   # (T, 256)

# build pool of all lv12 windows
print('embedding pool...', flush=True)
from tqdm import tqdm

pool_embs, pool_meta, pool_wins_raw = [], [], []

for _, row in tqdm(lv12.iterrows(), total=len(lv12)):
    if row['title'].lower() == 'quasar': continue
    npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
    if not npy.exists(): continue
    e, nc, _ = encode_chart_pretrain_v6(str(npy))
    p1w, _, _ = window_chart_split(e, nc, WINDOW_BARS, STRIDE_BARS)
    if p1w.shape[0] == 0: continue
    embs = embed_batch(p1w)
    raws = raw_windows(npy)
    T = len(embs)
    for t in range(T):
        pool_embs.append(embs[t])
        pool_meta.append({
            'title': row['title'],
            'win':   t + 1,
            'total': T,
            'pos_pct': f'{100*t/max(T-1,1):.0f}%',
            'ec': float(row['rating_ec']),
            'hc': float(row['rating_hc']),
        })
        pool_wins_raw.append(raws[t] if t < len(raws) else np.zeros((ROWS_PER_WIN, 16), dtype=np.uint8))

pool_embs = np.array(pool_embs)
pool_n    = pool_embs / (np.linalg.norm(pool_embs, axis=1, keepdims=True) + 1e-8)
q_n       = q_embs   / (np.linalg.norm(q_embs,    axis=1, keepdims=True) + 1e-8)

print(f'pool: {len(pool_embs)} windows', flush=True)

# ── Print windows 27–34 with top match ───────────────────────────────────────

for wi in WIN_RANGE:
    idx = wi - 1
    if idx >= len(q_wins_raw): break
    pos_pct = f'{100*idx/max(q_total-1,1):.0f}%'

    sims = pool_n @ q_n[idx]
    best = int(np.argmax(sims))
    sim  = float(sims[best])
    bm   = pool_meta[best]

    print(f'\n{"═"*76}')
    print(f'  QUASAR window {wi}/{q_total}  ({pos_pct})  ←→  '
          f'"{bm["title"]}" win {bm["win"]}/{bm["total"]} ({bm["pos_pct"]})  '
          f'sim={sim:.4f}')
    print(f'{"═"*76}')

    q_lines  = piano_roll(q_wins_raw[idx],     'quasar',       wi,       q_total, pos_pct, ec=float(q_row['rating_ec']))
    cmp_lines = piano_roll(pool_wins_raw[best], bm['title'], bm['win'], bm['total'], bm['pos_pct'], ec=bm['ec'])

    # print side by side
    max_l = max(len(q_lines), len(cmp_lines))
    q_lines   += [''] * (max_l - len(q_lines))
    cmp_lines += [''] * (max_l - len(cmp_lines))

    for a, b in zip(q_lines, cmp_lines):
        print(f'{a:<72}  {b}')
