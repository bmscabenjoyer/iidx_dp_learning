"""
Window-by-window nearest-neighbour search for quasar [DP ANOTHER] using the v7
BPM-agnostic time-based encoder. Windows are 8-second, 150 BPM reference.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from dataset import (
    encode_chart_pretrain_v6, window_chart_time_half,
    PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
    NUM_PATCHES_V7, PATCH_ROWS_V7,
    WIN_SECS, STRIDE_SECS,
)
from mae import MAEModel

MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
CKPT      = Path('checkpoints/pretrain_half_v7/encoder_best.pt')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 256
TOP_K     = 5


def load_encoder():
    model = MAEModel(
        in_channels=PRETRAIN_IN_CHANNELS_HALF,
        num_patches=NUM_PATCHES_V7, patch_rows=PATCH_ROWS_V7,
        encoder_dim=EMBED_DIM, decoder_dim=128, num_heads=8,
        use_lane_type_embed=True, use_bpm_cond=True,
        lane_types=LANE_TYPES_HALF,
    )
    model.encoder.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    return model.encoder.eval().to(DEVICE)


@torch.no_grad()
def embed_wins(encoder, wins):
    """wins: (T, WIN_ROWS_V7, 9)  → (T, 256)"""
    x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
    bpm = torch.from_numpy(wins[:, :, 8].mean(axis=1)).float().to(DEVICE)
    return encoder.embed_segment(x, bpm=bpm).cpu().numpy()


@torch.no_grad()
def embed_all(encoder, manifest):
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    all_embs, all_meta = [], []
    q_embs,   q_meta   = [], []

    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='embedding'):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        p1_wins, _, win_nc = window_chart_time_half(enc, nc)
        T = p1_wins.shape[0]
        if T == 0:
            continue
        embs = embed_wins(encoder, p1_wins)  # (T, 256)

        is_quasar = (isinstance(row['title'], str) and
                     row['title'].lower() == 'quasar')

        for t in range(T):
            meta = {
                'title':   row['title'],
                'pos_pct': f'{100*t/max(T-1,1):.0f}%',
                'win_idx': t,
                'total':   T,
                't_start': t * STRIDE_SECS,
                'bpm':     str(row['bpm']),
                'notes':   float(win_nc[t]),
                'ec':      float(row['rating_ec']),
                'hc':      float(row['rating_hc']),
            }
            all_embs.append(embs[t])
            all_meta.append(meta)
            if is_quasar:
                q_embs.append(embs[t])
                q_meta.append(meta)

    return (np.stack(all_embs), all_meta,
            np.stack(q_embs) if q_embs else None, q_meta)


# ── Main ──────────────────────────────────────────────────────────────────────

print(f'device: {DEVICE}', flush=True)
encoder  = load_encoder()
manifest = pd.read_csv(MANIFEST)

all_embs, all_meta, q_embs, q_meta = embed_all(encoder, manifest)

if q_embs is None:
    print('quasar not found!'); exit()

print(f'quasar: {len(q_meta)} windows  |  pool: {len(all_meta)} total', flush=True)

all_n = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
q_n   = q_embs   / (np.linalg.norm(q_embs,   axis=1, keepdims=True) + 1e-8)

print('\n' + '═'*100)
print(f'  quasar [DP ANOTHER]  ec={q_meta[0]["ec"]:.1f}  hc={q_meta[0]["hc"]:.1f}'
      f'  — window-by-window nearest neighbours (v7 BPM-agnostic encoder)')
print('═'*100)

header  = (f'  {"rank":>4}  {"sim":>6}  {"title":<40}  {"t":>7}  '
           f'{"bpm":>8}  {"notes":>6}  {"ec":>5}  {"hc":>5}')
divider = (f'  {"─"*4}  {"─"*6}  {"─"*40}  {"─"*7}  '
           f'{"─"*8}  {"─"*6}  {"─"*5}  {"─"*5}')

for wi, (q_emb_n, qm) in enumerate(zip(q_n, q_meta)):
    sims = all_n @ q_emb_n

    # mask quasar's own windows
    for i, m in enumerate(all_meta):
        if isinstance(m['title'], str) and m['title'].lower() == 'quasar':
            sims[i] = -1.0

    top = np.argsort(sims)[::-1][:TOP_K]

    t0 = qm['t_start']
    print(f'\n  ── win {wi+1:>2}/{len(q_meta)}  t={t0:.0f}–{t0+WIN_SECS:.0f}s'
          f'  ({qm["pos_pct"]:>4})  notes={qm["notes"]:.0f}')
    print(header)
    print(divider)
    for rank, j in enumerate(top, 1):
        m  = all_meta[j]
        t0m = m['t_start']
        print(f'  {rank:>4}  {sims[j]:>6.4f}  {m["title"]:<40}  '
              f'{t0m:.0f}–{t0m+WIN_SECS:.0f}s  '
              f'{m["bpm"]:>8}  {m["notes"]:>6.0f}  {m["ec"]:>5.1f}  {m["hc"]:>5.1f}')
