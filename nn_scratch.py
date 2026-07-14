"""
Nearest-neighbour retrieval focused on scratch-heavy charts.
Finds lv12 charts with high scratch lane (col 0 + col 15) density,
then shows their embedding neighbours to test if the encoder groups
scratch charts together structurally.
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
K_WEIGHT  = 5
TOP_K     = 10


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


@torch.no_grad()
def embed_chart(encoder, npy_path):
    enc, nc, _ = encode_chart_pretrain_v6(str(npy_path))
    p1_wins, p2_wins, _ = window_chart_split(enc, nc, WINDOW_BARS, STRIDE_BARS)
    if p1_wins.shape[0] == 0:
        return None
    def wmean(wins):
        x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
        bpm = torch.from_numpy(wins[:, :, 8].mean(1)).float().to(DEVICE)
        cls = encoder.embed_segment(x, bpm=bpm).cpu().numpy()
        T = len(cls)
        t = np.arange(T, dtype=np.float32)
        w = 1.0 + K_WEIGHT * (t / max(T - 1, 1)); w /= w.sum()
        return (cls * w[:, None]).sum(0)
    p1 = wmean(p1_wins); p2 = wmean(p2_wins)
    return np.concatenate([(p1 + p2) / 2.0, p1 - p2])


def scratch_ratio(npy_path):
    """Fraction of note events in scratch lanes (col 0 + col 15)."""
    arr = np.load(npy_path)          # (rows, 17)
    scratch = ((arr[:, 0] > 0) | (arr[:, 15] > 0)).sum()
    total   = (arr[:, :16] > 0).sum()
    return float(scratch) / max(total, 1)


# ── Build dataset ─────────────────────────────────────────────────────────────

print(f'device: {DEVICE}', flush=True)
encoder  = load_encoder()
manifest = pd.read_csv(MANIFEST)

# restrict to lv12 with ratings for the main pool
lv12 = manifest[
    (manifest['level'] == 12) &
    manifest['rating_ec'].notna() &
    (manifest['status'] == 'ok')
].reset_index(drop=True)

print('computing scratch ratios...', flush=True)
scratch_ratios = []
valid_idx = []
for i, row in tqdm(lv12.iterrows(), total=len(lv12)):
    npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
    if not npy.exists():
        scratch_ratios.append(0.0)
    else:
        scratch_ratios.append(scratch_ratio(npy))
    valid_idx.append(i)

lv12['scratch_ratio'] = scratch_ratios
top_scratch = lv12.sort_values('scratch_ratio', ascending=False).head(20)

print('\n── Top scratch-heavy lv12 charts ───────────────────────────────────────')
print(f'  {"title":<45} {"scratch%":>9}  {"ec":>6}  {"hc":>6}  {"bpm":>8}')
print(f'  {"─"*45} {"─"*9}  {"─"*6}  {"─"*6}  {"─"*8}')
for _, r in top_scratch.iterrows():
    print(f'  {r["title"]:<45} {r["scratch_ratio"]*100:>8.1f}%  '
          f'{r["rating_ec"]:>6.1f}  {r["rating_hc"]:>6.1f}  {str(r["bpm"]):>8}')

# ── Embed all lv12 charts ─────────────────────────────────────────────────────

print('\nembedding lv12 charts...', flush=True)
rows, embs = [], []
for _, row in tqdm(lv12.iterrows(), total=len(lv12)):
    npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
    if not npy.exists(): continue
    emb = embed_chart(encoder, npy)
    if emb is None: continue
    rows.append(row)
    embs.append(emb)

df   = pd.DataFrame(rows).reset_index(drop=True)
embs = np.stack(embs)
embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

def cosine_sim(a, B_n):
    a_n = a / (np.linalg.norm(a) + 1e-8)
    return B_n @ a_n

def show_neighbours(idx, label=''):
    query = df.iloc[idx]
    sims  = cosine_sim(embs_n[idx], embs_n)
    top   = np.argsort(sims)[::-1][:TOP_K]
    sr    = float(query['scratch_ratio']) * 100
    print(f'\nQuery: {query["title"]}  ec={query["rating_ec"]:.1f}  '
          f'bpm={query["bpm"]}  scratch={sr:.1f}%  {label}')
    print(f'  {"rank":>4}  {"sim":>6}  {"title":<45}  {"ec":>6}  {"hc":>6}  '
          f'{"bpm":>8}  {"scratch%":>9}')
    print(f'  {"─"*4}  {"─"*6}  {"─"*45}  {"─"*6}  {"─"*6}  {"─"*8}  {"─"*9}')
    for rank, j in enumerate(top, 1):
        r  = df.iloc[j]
        sr_j = float(r['scratch_ratio']) * 100
        tag = ' ←' if j == idx else ''
        print(f'  {rank:>4}  {sims[j]:>6.4f}  {r["title"]:<45}  '
              f'{r["rating_ec"]:>6.1f}  {r["rating_hc"]:>6.1f}  '
              f'{str(r["bpm"]):>8}  {sr_j:>8.1f}%{tag}')

# pick top 3 scratch-heavy charts that were successfully embedded
top_titles = top_scratch['title'].tolist()
query_indices = []
for t in top_titles:
    matches = df[df['title'] == t].index.tolist()
    if matches:
        query_indices.append(matches[0])
    if len(query_indices) == 3:
        break

print('\n── Nearest neighbours for scratch-heavy charts ──────────────────────────')
for qi in query_indices:
    show_neighbours(qi)

# also show a low-scratch chart for contrast
low_scratch_idx = df['scratch_ratio'].idxmin()
print('\n── Contrast: lowest-scratch chart ──────────────────────────────────────')
show_neighbours(low_scratch_idx, label='(low scratch)')
