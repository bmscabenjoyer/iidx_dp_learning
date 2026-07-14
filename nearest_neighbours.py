"""
Nearest-neighbour retrieval on pretrained half-chart encoder embeddings.

For each query chart, finds the top-K most similar charts in embedding space
(cosine similarity on k=5 weighted-mean [mean,diff] fused embeddings).
Structural similarity check: do similar embeddings correspond to similar patterns?
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

# ── Config ────────────────────────────────────────────────────────────────────
MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
CKPT      = Path('checkpoints/pretrain_half/encoder_best.pt')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 256
K_WEIGHT  = 5
TOP_K     = 7   # neighbours to show (includes self)

# ── Encoder ───────────────────────────────────────────────────────────────────

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

    def weighted_mean_cls(wins):
        x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
        bpm = torch.from_numpy(wins[:, :, 8].mean(1)).float().to(DEVICE)
        cls = encoder.embed_segment(x, bpm=bpm).cpu().numpy()  # (T, 256)
        T = len(cls)
        t = np.arange(T, dtype=np.float32)
        w = 1.0 + K_WEIGHT * (t / max(T - 1, 1))
        w /= w.sum()
        return (cls * w[:, None]).sum(0)  # (256,)

    p1 = weighted_mean_cls(p1_wins)
    p2 = weighted_mean_cls(p2_wins)
    return np.concatenate([(p1 + p2) / 2.0, p1 - p2])  # (512,)


# ── Pre-compute all chart embeddings ──────────────────────────────────────────

def precompute(encoder, manifest):
    rows, embs = [], []
    skipped = 0
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc='embedding'):
        # resolve path relative to level directory
        level = int(row['level'])
        npy = DATA_ROOT / f'dp{level}_active' / str(row['file_path'])
        if not npy.exists():
            skipped += 1
            continue
        emb = embed_chart(encoder, npy)
        if emb is None:
            skipped += 1
            continue
        rows.append(row)
        embs.append(emb)
    print(f'embedded {len(embs)} charts  (skipped {skipped})', flush=True)
    return pd.DataFrame(rows).reset_index(drop=True), np.stack(embs)


def cosine_sim(a, B):
    """Cosine similarity between vector a (512,) and matrix B (N, 512)."""
    a_n = a / (np.linalg.norm(a) + 1e-8)
    B_n = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-8)
    return B_n @ a_n


def show_neighbours(idx, df, embs, top_k=TOP_K):
    query = df.iloc[idx]
    sims  = cosine_sim(embs[idx], embs)
    top   = np.argsort(sims)[::-1][:top_k]

    label = (f"  ec={query['rating_ec']:.1f}" if pd.notna(query.get('rating_ec')) else '')
    print(f"\nQuery [{idx}]: {query['title']}  lv{int(query['level'])}  {query['diftype']}{label}")
    print(f"  bpm={query['bpm']}  notes={int(query['num_notes']) if pd.notna(query['num_notes']) else '?'}")
    print(f"  {'rank':>4}  {'sim':>6}  {'title':<40}  {'lv':>3}  {'diftype':<16}  {'ec':>6}  {'hc':>6}  {'bpm':>8}  {'notes':>6}")
    print(f"  {'─'*4}  {'─'*6}  {'─'*40}  {'─'*3}  {'─'*16}  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*6}")
    for rank, j in enumerate(top, 1):
        r   = df.iloc[j]
        ec  = f"{r['rating_ec']:.1f}"  if pd.notna(r.get('rating_ec'))  else '  —  '
        hc  = f"{r['rating_hc']:.1f}"  if pd.notna(r.get('rating_hc'))  else '  —  '
        tag = ' ← query' if j == idx else ''
        print(f"  {rank:>4}  {sims[j]:>6.4f}  {r['title']:<40}  {int(r['level']):>3}  "
              f"{r['diftype']:<16}  {ec:>6}  {hc:>6}  {str(r['bpm']):>8}  "
              f"{int(r['num_notes']) if pd.notna(r['num_notes']) else 0:>6}{tag}")


# ── Main ──────────────────────────────────────────────────────────────────────

print(f'device: {DEVICE}', flush=True)
encoder  = load_encoder()
manifest = pd.read_csv(MANIFEST)
# use all levels so lv12 queries can find lv10/11 structural neighbours too
df, embs = precompute(encoder, manifest)

# normalise embeddings for cosine sim
embs_n = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-8)

# ── Query selection: pick a spread of lv12 charts ────────────────────────────
lv12_mask = (df['level'] == 12) & df['rating_ec'].notna()
lv12_idx  = df[lv12_mask].index.tolist()

# hardest EC, easiest EC, median, and two randoms
rated     = df.loc[lv12_idx].copy()
rated['_i'] = lv12_idx
rated_s   = rated.sort_values('rating_ec')

queries = [
    rated_s.iloc[0]['_i'],                          # easiest EC
    rated_s.iloc[len(rated_s)//4]['_i'],            # 25th pct
    rated_s.iloc[len(rated_s)//2]['_i'],            # median
    rated_s.iloc[3*len(rated_s)//4]['_i'],          # 75th pct
    rated_s.iloc[-1]['_i'],                         # hardest EC
]

for q in queries:
    show_neighbours(int(q), df, embs)
