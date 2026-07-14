"""
Window-level nearest-neighbour search.

Embeds every individual window from every lv12 chart.
For a query window, finds the most similar windows across all charts,
reporting which song and where in that song (position %) the match came from.
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
TOP_K     = 8


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
    """Scratch note fraction for a single window array (rows, 9)."""
    notes = win[:, :8]  # lanes only
    scratch = ((notes[:, 0] > 0)).sum()   # P1 scratch lane (col 0 in half-enc)
    total   = (notes > 0).sum()
    return float(scratch) / max(total, 1)


@torch.no_grad()
def embed_all_windows(encoder, manifest):
    """
    Returns:
        all_embs   : (N_windows, 256) — per-window P1 CLS embeddings
        all_meta   : list of dicts with title, pos_frac, bpm, scratch_ratio, rating_ec, rating_hc
    """
    all_embs, all_meta = [], []
    skipped = 0

    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='embedding windows'):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            skipped += 1; continue

        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        p1_wins, p2_wins, win_nc = window_chart_split(enc, nc, WINDOW_BARS, STRIDE_BARS)
        T = p1_wins.shape[0]
        if T == 0:
            skipped += 1; continue

        # embed P1 windows (256-dim each)
        x   = torch.from_numpy(p1_wins[:, :, :8]).float().to(DEVICE)
        bpm = torch.from_numpy(p1_wins[:, :, 8].mean(1)).float().to(DEVICE)
        cls = encoder.embed_segment(x, bpm=bpm).cpu().numpy()  # (T, 256)

        for t in range(T):
            sr = scratch_ratio_window(p1_wins[t])
            all_embs.append(cls[t])
            all_meta.append({
                'title':      row['title'],
                'diftype':    row['diftype'],
                'pos_frac':   t / max(T - 1, 1),   # 0.0 = start, 1.0 = end
                'pos_pct':    f'{100*t/max(T-1,1):.0f}%',
                'window_idx': t,
                'total_wins': T,
                'bpm':        str(row['bpm']),
                'scratch':    sr,
                'rating_ec':  float(row['rating_ec']),
                'rating_hc':  float(row['rating_hc']),
                'win_nc':     float(win_nc[t]),
            })

    print(f'embedded {len(all_embs)} windows from {len(lv12)-skipped} charts '
          f'(skipped {skipped})', flush=True)
    return np.stack(all_embs), all_meta


def show_window_neighbours(q_idx, embs_n, meta, top_k=TOP_K, label=''):
    q   = meta[q_idx]
    sim = embs_n @ embs_n[q_idx]
    top = np.argsort(sim)[::-1][:top_k]

    print(f'\nQuery window: "{q["title"]}"  {q["diftype"]}')
    print(f'  position {q["pos_pct"]} through song  '
          f'(win {q["window_idx"]+1}/{q["total_wins"]})  '
          f'bpm={q["bpm"]}  scratch={q["scratch"]*100:.1f}%  '
          f'notes={q["win_nc"]:.0f}  ec={q["rating_ec"]:.1f}  {label}')
    print(f'  {"rank":>4}  {"sim":>6}  {"title":<42}  {"pos":>5}  '
          f'{"bpm":>8}  {"scratch%":>9}  {"notes":>6}  {"ec":>5}  {"hc":>5}')
    print(f'  {"─"*4}  {"─"*6}  {"─"*42}  {"─"*5}  '
          f'{"─"*8}  {"─"*9}  {"─"*6}  {"─"*5}  {"─"*5}')
    for rank, j in enumerate(top, 1):
        m   = meta[j]
        tag = ' ←' if j == q_idx else ''
        print(f'  {rank:>4}  {sim[j]:>6.4f}  {m["title"]:<42}  {m["pos_pct"]:>5}  '
              f'{m["bpm"]:>8}  {m["scratch"]*100:>8.1f}%  {m["win_nc"]:>6.0f}  '
              f'{m["rating_ec"]:>5.1f}  {m["rating_hc"]:>5.1f}{tag}')


# ── Main ──────────────────────────────────────────────────────────────────────

print(f'device: {DEVICE}', flush=True)
encoder  = load_encoder()
manifest = pd.read_csv(MANIFEST)

all_embs, meta = embed_all_windows(encoder, manifest)
embs_n = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
print(f'total windows: {len(meta)}', flush=True)

# ── Query selection ───────────────────────────────────────────────────────────

# 1. Hardest window of the top scratch chart (Level 2)
# 2. Final window of the hardest chart (quell)
# 3. Most scratch-heavy individual window in the dataset
# 4. Dense window from a high-BPM chart

meta_arr = pd.DataFrame(meta)

# Hardest-rated scratch-heavy chart
level2_idx = meta_arr[meta_arr['title'] == 'Level 2']
if not level2_idx.empty:
    # pick the window with most scratch content
    q1 = level2_idx.loc[level2_idx['scratch'].idxmax()].name
    show_window_neighbours(q1, embs_n, meta, label='(most scratch-heavy window)')

# Final window of quell (hardest ec)
quell_idx = meta_arr[meta_arr['title'].str.contains('quell', case=False)]
if not quell_idx.empty:
    q2 = quell_idx.loc[quell_idx['pos_frac'].idxmax()].name
    show_window_neighbours(q2, embs_n, meta, label='(final window)')

# Single window with highest scratch ratio in entire dataset
q3 = meta_arr['scratch'].idxmax()
show_window_neighbours(int(q3), embs_n, meta, label='(max scratch window in dataset)')

# Dense window from highest BPM chart
high_bpm = meta_arr[meta_arr['bpm'].str.match(r'^\d+$', na=False)]
high_bpm = high_bpm[high_bpm['bpm'].astype(int) >= 200]
if not high_bpm.empty:
    q4 = high_bpm.loc[high_bpm['win_nc'].idxmax()].name
    show_window_neighbours(int(q4), embs_n, meta, label='(densest window at 200+ bpm)')
