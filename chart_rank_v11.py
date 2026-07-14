"""
Non-overlapping window sweep + chart-level ranking using the v11 encoder.
Sweeps the query chart in non-overlapping 4s windows (stride = WIN_SECS_V10),
finds top-5 matches per window, then aggregates into a chart ranking by
rank-weighted score (sum of 1/rank per appearance).

P1+P2 concatenated embeddings (384-dim). Flip-invariant search.
"""

import argparse
import collections
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

from dataset import (
    encode_chart_pretrain_v6, window_chart_time_half,
    PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
    NUM_PATCHES_V10, PATCH_ROWS_V7,
    WIN_SECS_V10, WIN_ROWS_V10,
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
    q_nonoverlap = []
    q_win_nc     = None

    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='embedding pool', file=__import__('sys').stderr):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        # overlapping windows for pool (full coverage)
        p1_wins, p2_wins, win_nc = window_chart_time_half(
            enc, nc, win_secs=WIN_SECS_V10, stride_secs=WIN_SECS_V10 / 2,
            win_rows=WIN_ROWS_V10,
        )
        T = p1_wins.shape[0]
        if T == 0:
            continue
        p1_embs = embed_wins(encoder, p1_wins)
        p2_embs = embed_wins(encoder, p2_wins)
        embs    = np.concatenate([p1_embs, p2_embs], axis=1)  # (T, 384)

        title    = row['title']   if isinstance(row['title'],   str) else ''
        diftype  = row['diftype'] if isinstance(row['diftype'], str) else ''
        is_query = (title.lower() == query_title.lower() and
                    diftype.lower() == query_diftype.lower())

        chart_key = (title, diftype)
        for t in range(T):
            meta = {
                'title':     title,
                'diftype':   diftype,
                'chart_key': chart_key,
                't_start':   t * (WIN_SECS_V10 / 2),
                'ec':        float(row['rating_ec']),
                'hc':        float(row['rating_hc']) if pd.notna(row['rating_hc']) else 0.0,
                'is_query':  is_query,
            }
            all_embs.append(embs[t])
            all_meta.append(meta)
            if is_query:
                q_embs.append(embs[t])
                q_meta.append(meta)

        # query windows with 2s stride (overlapping) for finer alignment
        if is_query:
            p1_no, p2_no, nc_no = window_chart_time_half(
                enc, nc, win_secs=WIN_SECS_V10, stride_secs=WIN_SECS_V10 / 2,
                win_rows=WIN_ROWS_V10,
            )
            if p1_no.shape[0] > 0:
                p1_no_e = embed_wins(encoder, p1_no)
                p2_no_e = embed_wins(encoder, p2_no)
                q_nonoverlap = np.concatenate([p1_no_e, p2_no_e], axis=1)
                q_win_nc = nc_no  # note counts per non-overlapping window

    # fallback: query not in rated pool — find it anywhere in lv12
    if not len(q_nonoverlap):
        lv12_full = manifest[
            (manifest['level'] == 12) &
            (manifest['status'] == 'ok')
        ]
        q_rows = lv12_full[
            (lv12_full['title'].str.lower()   == query_title.lower()) &
            (lv12_full['diftype'].str.lower()  == query_diftype.lower())
        ]
        for _, row in q_rows.iterrows():
            npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
            if not npy.exists():
                continue
            enc, nc, _ = encode_chart_pretrain_v6(str(npy))
            p1_no, p2_no, nc_no = window_chart_time_half(
                enc, nc, win_secs=WIN_SECS_V10, stride_secs=WIN_SECS_V10 / 2,
                win_rows=WIN_ROWS_V10,
            )
            if p1_no.shape[0] > 0:
                p1_no_e = embed_wins(encoder, p1_no)
                p2_no_e = embed_wins(encoder, p2_no)
                q_nonoverlap = np.concatenate([p1_no_e, p2_no_e], axis=1)
                q_win_nc     = nc_no
            break

    return (np.stack(all_embs), all_meta,
            np.stack(q_embs) if q_embs else None, q_meta,
            q_nonoverlap if len(q_nonoverlap) else None,
            q_win_nc)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--query',   required=True)
    p.add_argument('--diftype', default='[DP ANOTHER]')
    p.add_argument('--top-k',   type=int, default=TOP_K)
    p.add_argument('--top-charts', type=int, default=20,
                   help='number of top charts to display in ranking')
    p.add_argument('--density-exp', type=float, default=2.0,
                   help='exponent applied to note counts before normalising weights '
                        '(1.0=linear, 2.0=quadratic, higher=more focus on dense windows)')
    args = p.parse_args()

    print(f'device: {DEVICE}', flush=True)
    encoder  = load_encoder()
    manifest = pd.read_csv(MANIFEST)

    all_embs, all_meta, q_embs, q_meta, q_embs_full, q_win_nc = embed_all(
        encoder, manifest, args.query, args.diftype)

    if q_embs_full is None:
        print(f'"{args.query}" {args.diftype} not found in lv12 charts!')
        return
    if q_embs is None:
        print(f'  (note: query has no ereter rating — not in pool, results are unfiltered)')


    T_q = q_embs_full.shape[0]

    # density weights: raise note counts to density_exp, then mean-normalise
    # so the relative contribution of dense vs sparse windows scales as nc^exp
    nc_arr  = np.array(q_win_nc, dtype=float)
    powered = nc_arr ** args.density_exp
    weights = powered / (powered.mean() + 1e-8)

    all_n    = all_embs / (np.linalg.norm(all_embs, axis=1, keepdims=True) + 1e-8)
    q_n      = q_embs_full / (np.linalg.norm(q_embs_full, axis=1, keepdims=True) + 1e-8)
    half     = EMBED_DIM
    q_flip   = np.concatenate([q_embs_full[:, half:], q_embs_full[:, :half]], axis=1)
    q_flip_n = q_flip / (np.linalg.norm(q_flip, axis=1, keepdims=True) + 1e-8)

    # aggregate scores per chart
    scores    = collections.defaultdict(float)
    hit_count = collections.defaultdict(int)
    best_sim  = collections.defaultdict(float)
    chart_info = {}

    for wi in range(T_q):
        w    = float(weights[wi])
        sims = np.maximum(all_n @ q_n[wi], all_n @ q_flip_n[wi])
        if q_embs is not None:
            for i, m in enumerate(all_meta):
                if m['is_query']:
                    sims[i] = -1.0
        top = np.argsort(sims)[::-1][:args.top_k]
        for rank, j in enumerate(top, 1):
            m   = all_meta[j]
            key = m['chart_key']
            scores[key]    += w / rank
            hit_count[key] += 1
            best_sim[key]   = max(best_sim[key], float(sims[j]))
            if key not in chart_info:
                chart_info[key] = {'ec': m['ec'], 'hc': m['hc'],
                                   'title': m['title'], 'diftype': m['diftype']}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    qm0_rows = manifest[
        (manifest['title'].str.lower() == args.query.lower()) &
        (manifest['diftype'].str.lower() == args.diftype.lower())
    ]
    qm0_row  = qm0_rows.iloc[0] if len(qm0_rows) else None
    qm0_ec   = f'{float(qm0_row["rating_ec"]):.1f}' if (qm0_row is not None and pd.notna(qm0_row['rating_ec'])) else 'N/A'
    qm0_hc   = f'{float(qm0_row["rating_hc"]):.1f}' if (qm0_row is not None and pd.notna(qm0_row['rating_hc'])) else 'N/A'

    print(f'\n{args.query.upper()} {args.diftype}  ec={qm0_ec}  hc={qm0_hc}'
          f'  — {T_q} overlapping {WIN_SECS_V10:.0f}s windows (2s stride)  |  pool: {len(all_meta)} windows')
    print(f'  density exp={args.density_exp}  weights: min={weights.min():.2f}  mean=1.00  max={weights.max():.2f}')
    print('\n' + '═'*110)
    print(f'  Chart-level ranking by density-weighted score  (score = Σ density_weight/rank over {T_q} windows)')
    print('═'*110)
    header  = (f'  {"rank":>4}  {"score":>6}  {"hits":>4}  {"best_sim":>8}  '
               f'{"title":<44}  {"diftype":<16}  {"ec":>5}  {"hc":>5}')
    divider = (f'  {"─"*4}  {"─"*6}  {"─"*4}  {"─"*8}  '
               f'{"─"*44}  {"─"*16}  {"─"*5}  {"─"*5}')
    print(header)
    print(divider)
    for rank, (key, score) in enumerate(ranked[:args.top_charts], 1):
        info = chart_info[key]
        print(f'  {rank:>4}  {score:>6.2f}  {hit_count[key]:>4}  {best_sim[key]:>8.4f}  '
              f'{info["title"]:<44}  {info["diftype"]:<16}  '
              f'{info["ec"]:>5.1f}  {info["hc"]:>5.1f}')


if __name__ == '__main__':
    main()
