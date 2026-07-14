"""
Window-by-window nearest-neighbour search for quasar [DP ANOTHER] across all lv12.
For each window in quasar, shows the top-5 most similar windows from OTHER charts.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

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
TOP_K     = 5


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


def scratch_ratio_window(win):
    notes = win[:, :8]
    scratch = (notes[:, 0] > 0).sum()
    total   = (notes > 0).sum()
    return float(scratch) / max(total, 1)


@torch.no_grad()
def embed_all_windows(encoder, manifest):
    all_embs, all_meta = [], []

    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    quasar_embs, quasar_meta = [], []

    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='embedding'):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists(): continue

        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        p1_wins, p2_wins, win_nc = window_chart_split(enc, nc, WINDOW_BARS, STRIDE_BARS)
        T = p1_wins.shape[0]
        if T == 0: continue

        x   = torch.from_numpy(p1_wins[:, :, :8]).float().to(DEVICE)
        bpm = torch.from_numpy(p1_wins[:, :, 8].mean(1)).float().to(DEVICE)
        cls = encoder.embed_segment(x, bpm=bpm).cpu().numpy()  # (T, 256)

        is_quasar = (isinstance(row['title'], str) and
                     row['title'].lower() == 'quasar' and
                     '[DP ANOTHER]' in str(row['diftype']))

        for t in range(T):
            meta = {
                'title':     row['title'],
                'diftype':   row['diftype'],
                'pos_frac':  t / max(T - 1, 1),
                'pos_pct':   f'{100*t/max(T-1,1):.0f}%',
                'win_idx':   t,
                'total':     T,
                'bpm':       str(row['bpm']),
                'scratch':   scratch_ratio_window(p1_wins[t]),
                'notes':     float(win_nc[t]),
                'ec':        float(row['rating_ec']),
                'hc':        float(row['rating_hc']),
            }
            all_embs.append(cls[t])
            all_meta.append(meta)
            if is_quasar:
                quasar_embs.append(cls[t])
                quasar_meta.append(meta)

    return (np.stack(all_embs), all_meta,
            np.stack(quasar_embs) if quasar_embs else None, quasar_meta)


# ── Main ──────────────────────────────────────────────────────────────────────

print(f'device: {DEVICE}', flush=True)
encoder  = load_encoder()
manifest = pd.read_csv(MANIFEST)

all_embs, all_meta, q_embs, q_meta = embed_all_windows(encoder, manifest)

if q_embs is None:
    print('quasar not found!'); exit()

print(f'quasar has {len(q_meta)} windows  |  pool: {len(all_meta)} total windows',
      flush=True)

# normalise
all_n = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
q_n   = q_embs   / (np.linalg.norm(q_embs,   axis=1, keepdims=True) + 1e-8)

print('\n' + '═'*100)
print(f'  quasar [DP ANOTHER]  lv12  ec={q_meta[0]["ec"]:.1f}  hc={q_meta[0]["hc"]:.1f}  '
      f'bpm={q_meta[0]["bpm"]}  — window-by-window nearest neighbours')
print('═'*100)

header = (f'  {"rank":>4}  {"sim":>6}  {"title":<40}  {"pos":>5}  '
          f'{"bpm":>8}  {"scratch%":>9}  {"notes":>6}  {"ec":>5}  {"hc":>5}')
divider = (f'  {"─"*4}  {"─"*6}  {"─"*40}  {"─"*5}  '
           f'{"─"*8}  {"─"*9}  {"─"*6}  {"─"*5}  {"─"*5}')

for wi, (q_emb_n, qm) in enumerate(zip(q_n, q_meta)):
    sims = all_n @ q_emb_n                    # (N_all,)

    # mask out quasar's own windows
    for i, m in enumerate(all_meta):
        if isinstance(m['title'], str) and m['title'].lower() == 'quasar':
            sims[i] = -1.0

    top = np.argsort(sims)[::-1][:TOP_K]

    print(f'\n  ── window {wi+1:>2}/{len(q_meta)}  ({qm["pos_pct"]:>4} through song)  '
          f'notes={qm["notes"]:.0f}  scratch={qm["scratch"]*100:.0f}%')
    print(header)
    print(divider)
    for rank, j in enumerate(top, 1):
        m = all_meta[j]
        print(f'  {rank:>4}  {sims[j]:>6.4f}  {m["title"]:<40}  {m["pos_pct"]:>5}  '
              f'{m["bpm"]:>8}  {m["scratch"]*100:>8.1f}%  {m["notes"]:>6.0f}  '
              f'{m["ec"]:>5.1f}  {m["hc"]:>5.1f}')
