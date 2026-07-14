"""
Window-by-window nearest-neighbour search using the v11 encoder.
embed_dim=192, depth=4, RoPE, 4s BPM-agnostic windows, 2s stride.
P1+P2 concatenated embeddings (384-dim). Query searched in both orientations
([p1,p2] and [p2,p1]) for flip-invariance; best similarity taken per db window.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from dataset import (
    encode_chart_pretrain_v6, window_chart_time_half_v10,
    PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
    NUM_PATCHES_V10, PATCH_ROWS_V7,
    WIN_SECS_V10, STRIDE_SECS_V10,
)
from mae import MAEModel

MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
CKPT      = Path('checkpoints/pretrain_half_v11/encoder_best.pt')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 192
TOP_K     = 5


def load_encoder():
    model = MAEModel(
        in_channels=PRETRAIN_IN_CHANNELS_HALF,
        num_patches=NUM_PATCHES_V10, patch_rows=PATCH_ROWS_V7,
        encoder_dim=EMBED_DIM, decoder_dim=96, num_heads=4,
        encoder_depth=4, use_rope=True,
        use_lane_type_embed=True, use_bpm_cond=False,
        lane_types=LANE_TYPES_HALF,
    )
    model.encoder.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    return model.encoder.eval().to(DEVICE)


@torch.no_grad()
def embed_wins(encoder, wins):
    x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
    bpm = torch.from_numpy(wins[:, :, 8].mean(axis=1)).float().to(DEVICE)
    return encoder.embed_segment(x, bpm=bpm).cpu().numpy()  # (T, 192)


@torch.no_grad()
def embed_all(encoder, manifest, query_title, query_diftype):
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    all_embs, all_meta = [], []
    q_embs,   q_meta   = [], []

    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='embedding pool'):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        p1_wins, p2_wins, win_nc = window_chart_time_half_v10(enc, nc)
        T = p1_wins.shape[0]
        if T == 0:
            continue
        p1_embs = embed_wins(encoder, p1_wins)   # (T, 192)
        p2_embs = embed_wins(encoder, p2_wins)   # (T, 192)
        embs    = np.concatenate([p1_embs, p2_embs], axis=1)  # (T, 384)

        title    = row['title']   if isinstance(row['title'],   str) else ''
        diftype  = row['diftype'] if isinstance(row['diftype'], str) else ''
        is_query = (title.lower() == query_title.lower() and
                    diftype.lower() == query_diftype.lower())

        for t in range(T):
            meta = {
                'title':    title,
                'diftype':  diftype,
                'pos_pct':  f'{100*t/max(T-1,1):.0f}%',
                'win_idx':  t,
                'total':    T,
                't_start':  t * STRIDE_SECS_V10,
                'bpm':      str(row['bpm']),
                'notes':    float(win_nc[t]),
                'ec':       float(row['rating_ec']),
                'hc':       float(row['rating_hc']) if pd.notna(row['rating_hc']) else 0.0,
                'is_query': is_query,
            }
            all_embs.append(embs[t])
            all_meta.append(meta)
            if is_query:
                q_embs.append(embs[t])
                q_meta.append(meta)

    return (np.stack(all_embs), all_meta,
            np.stack(q_embs) if q_embs else None, q_meta)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query',   required=True)
    p.add_argument('--diftype', default='[DP ANOTHER]')
    p.add_argument('--top-k',   type=int, default=TOP_K)
    p.add_argument('--dense-frac', type=float, default=0.25,
                   help='show only windows in the top dense_frac by note count')
    args = p.parse_args()

    print(f'device: {DEVICE}', flush=True)
    encoder  = load_encoder()
    manifest = pd.read_csv(MANIFEST)

    all_embs, all_meta, q_embs, q_meta = embed_all(
        encoder, manifest, args.query, args.diftype)

    if q_embs is None:
        print(f'"{args.query}" {args.diftype} not found in lv12 labeled pool!')
        return

    # filter query windows to densest fraction
    notes_arr = np.array([m['notes'] for m in q_meta])
    threshold = np.quantile(notes_arr, 1.0 - args.dense_frac)
    dense_idx = [i for i, m in enumerate(q_meta) if m['notes'] >= threshold]

    qm0 = q_meta[0]
    print(f'\n{args.query.upper()} {args.diftype}: {len(q_meta)} windows  '
          f'(showing {len(dense_idx)} densest)  |  pool: {len(all_meta)} total')

    all_n  = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
    q_n    = q_embs   / (np.linalg.norm(q_embs,   axis=1, keepdims=True) + 1e-8)
    # flipped orientation: swap p1 and p2 halves
    half   = EMBED_DIM
    q_flip = np.concatenate([q_embs[:, half:], q_embs[:, :half]], axis=1)
    q_flip_n = q_flip / (np.linalg.norm(q_flip, axis=1, keepdims=True) + 1e-8)

    print('\n' + '═'*105)
    print(f'  {args.query.upper()} {args.diftype}  ec={qm0["ec"]:.1f}  hc={qm0["hc"]:.1f}'
          f'  — densest windows, nearest neighbours (v11 encoder)')
    print('═'*105)

    header  = (f'  {"win":>4}  {"rank":>4}  {"sim":>6}  {"title":<40}  {"t":>7}  '
               f'{"bpm":>8}  {"notes":>6}  {"ec":>5}  {"hc":>5}')
    divider = (f'  {"─"*4}  {"─"*4}  {"─"*6}  {"─"*40}  {"─"*7}  '
               f'{"─"*8}  {"─"*6}  {"─"*5}  {"─"*5}')

    for wi in dense_idx:
        qm = q_meta[wi]
        # take max similarity across both orientations
        sims = np.maximum(all_n @ q_n[wi], all_n @ q_flip_n[wi])
        for i, m in enumerate(all_meta):
            if m['is_query']:
                sims[i] = -1.0
        top = np.argsort(sims)[::-1][:args.top_k]

        t0 = qm['t_start']
        print(f'\n  ── win {wi+1:>2}/{len(q_meta)}  t={t0:.0f}–{t0+WIN_SECS_V10:.0f}s'
              f'  ({qm["pos_pct"]:>4})  notes={qm["notes"]:.0f}')
        print(header)
        print(divider)
        for rank, j in enumerate(top, 1):
            m   = all_meta[j]
            t0m = m['t_start']
            print(f'  {wi+1:>4}  {rank:>4}  {sims[j]:>6.4f}  {m["title"]:<40}  '
                  f'{t0m:.0f}–{t0m+WIN_SECS_V10:.0f}s  '
                  f'{m["bpm"]:>8}  {m["notes"]:>6.0f}  {m["ec"]:>5.1f}  {m["hc"]:>5.1f}')


if __name__ == '__main__':
    main()
