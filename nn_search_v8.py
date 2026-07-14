"""
Window-by-window nearest-neighbour search using the v8 encoder.
embed_dim=384, depth=4, uniform masking, 8s BPM-agnostic windows.

Usage:
    python nn_search_v8.py --query "quasar"
    python nn_search_v8.py --query "LIGHTNING STRIKES"
"""

import argparse
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
CKPT      = Path('checkpoints/pretrain_half_v8/encoder_best.pt')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 384
TOP_K     = 5


def load_encoder():
    model = MAEModel(
        in_channels=PRETRAIN_IN_CHANNELS_HALF,
        num_patches=NUM_PATCHES_V7, patch_rows=PATCH_ROWS_V7,
        encoder_dim=EMBED_DIM, decoder_dim=128, num_heads=6,
        encoder_depth=4,
        use_lane_type_embed=True, use_bpm_cond=True,
        lane_types=LANE_TYPES_HALF,
    )
    model.encoder.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    return model.encoder.eval().to(DEVICE)


@torch.no_grad()
def embed_wins(encoder, wins):
    x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
    bpm = torch.from_numpy(wins[:, :, 8].mean(axis=1)).float().to(DEVICE)
    return encoder.embed_segment(x, bpm=bpm).cpu().numpy()


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
        p1_wins, _, win_nc = window_chart_time_half(enc, nc)
        T = p1_wins.shape[0]
        if T == 0:
            continue
        embs = embed_wins(encoder, p1_wins)

        title   = row['title']   if isinstance(row['title'],   str) else ''
        diftype = row['diftype'] if isinstance(row['diftype'], str) else ''
        is_query = (title.lower() == query_title.lower() and
                    diftype.lower() == query_diftype.lower())
        # mask key: only exclude windows that are the exact query chart
        is_query_chart = is_query

        for t in range(T):
            meta = {
                'title':    title,
                'diftype':  diftype,
                'pos_pct':  f'{100*t/max(T-1,1):.0f}%',
                'win_idx':  t,
                'total':    T,
                't_start':  t * STRIDE_SECS,
                'bpm':      str(row['bpm']),
                'notes':    float(win_nc[t]),
                'ec':       float(row['rating_ec']),
                'hc':       float(row['rating_hc']) if pd.notna(row['rating_hc']) else 0.0,
                'is_query': is_query_chart,
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
    p.add_argument('--query',   required=True, help='song title to search from (case-insensitive)')
    p.add_argument('--diftype', default='[DP ANOTHER]', help='diftype of query chart (default: [DP ANOTHER])')
    p.add_argument('--top-k',   type=int, default=TOP_K)
    args = p.parse_args()

    print(f'device: {DEVICE}', flush=True)
    encoder  = load_encoder()
    manifest = pd.read_csv(MANIFEST)

    all_embs, all_meta, q_embs, q_meta = embed_all(encoder, manifest, args.query, args.diftype)

    if q_embs is None:
        print(f'"{args.query}" {args.diftype} not found in lv12 labeled pool!')
        return

    qm0 = q_meta[0]
    print(f'\n{args.query.upper()} {args.diftype}: {len(q_meta)} windows  |  pool: {len(all_meta)} total')

    all_n = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
    q_n   = q_embs   / (np.linalg.norm(q_embs,   axis=1, keepdims=True) + 1e-8)

    print('\n' + '═'*105)
    print(f'  {args.query.upper()} {args.diftype}  ec={qm0["ec"]:.1f}  hc={qm0["hc"]:.1f}'
          f'  — window-by-window nearest neighbours (v8 encoder)')
    print('═'*105)

    header  = (f'  {"rank":>4}  {"sim":>6}  {"title":<40}  {"t":>7}  '
               f'{"bpm":>8}  {"notes":>6}  {"ec":>5}  {"hc":>5}')
    divider = (f'  {"─"*4}  {"─"*6}  {"─"*40}  {"─"*7}  '
               f'{"─"*8}  {"─"*6}  {"─"*5}  {"─"*5}')

    for wi, (q_emb_n, qm) in enumerate(zip(q_n, q_meta)):
        sims = all_n @ q_emb_n

        # mask only the exact query chart, not variants (e.g. DPL of the same song)
        for i, m in enumerate(all_meta):
            if m['is_query']:
                sims[i] = -1.0

        top = np.argsort(sims)[::-1][:args.top_k]

        t0 = qm['t_start']
        print(f'\n  ── win {wi+1:>2}/{len(q_meta)}  t={t0:.0f}–{t0+WIN_SECS:.0f}s'
              f'  ({qm["pos_pct"]:>4})  notes={qm["notes"]:.0f}')
        print(header)
        print(divider)
        for rank, j in enumerate(top, 1):
            m   = all_meta[j]
            t0m = m['t_start']
            print(f'  {rank:>4}  {sims[j]:>6.4f}  {m["title"]:<40}  '
                  f'{t0m:.0f}–{t0m+WIN_SECS:.0f}s  '
                  f'{m["bpm"]:>8}  {m["notes"]:>6.0f}  {m["ec"]:>5.1f}  {m["hc"]:>5.1f}')


if __name__ == '__main__':
    main()
