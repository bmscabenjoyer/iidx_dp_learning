"""
K-means archetype clustering on v11 encoder segment embeddings.

Each 4-second window is encoded as a 384-dim fused vector:
  [(p1+p2)/2 , p1-p2]  — symmetric pattern mean + left/right asymmetry.

1. Embed all charts (all levels) at the segment level.
2. K-means on L2-normalised embeddings.
3. Print top-N representative segments per cluster (with optional ASCII).
4. Per-chart archetype histograms → Jensen-Shannon nearest-neighbour retrieval.
5. Signature segments: rarest cluster membership or max centroid distance.
"""

import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import jensenshannon

from dataset import (
    encode_chart_pretrain_v6, window_chart_time_half_v10,
    PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
    NUM_PATCHES_V10, PATCH_ROWS_V7,
    NUM_PATCHES_V12, PATCH_ROWS_V12,
    ROWS_PER_BAR, BPM_SCALE,
    WIN_SECS_V10, STRIDE_SECS_V10,
)
from mae import MAEModel

# ── Config ─────────────────────────────────────────────────────────────────────
MANIFEST   = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT  = Path('/home/jysuh/projects/iidx_data')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEVEL_DIRS = {10: 'dp10_active', 11: 'dp11_active', 12: 'dp12_active'}

HALF_CH   = 8        # lanes per hand (scratch + key 1–7)
EMBED_DIM = 192      # per-hand CLS dim
FUSE_DIM  = 384      # [(p1+p2)/2 , p1-p2]
FUSE_DIM_TD = 768    # temporal-diff mode: above + [mean_diff_sym, mean_diff_asym]

ENCODER_CONFIGS = {
    'v11': dict(num_patches=NUM_PATCHES_V10, patch_rows=PATCH_ROWS_V7,
                ckpt='checkpoints/pretrain_half_v11/encoder_best.pt'),
    'v12': dict(num_patches=NUM_PATCHES_V12, patch_rows=PATCH_ROWS_V12,
                ckpt='checkpoints/pretrain_half_v12/encoder_best.pt'),
}


# ── Encoder ───────────────────────────────────────────────────────────────────

def load_encoder(version: str = 'v12'):
    cfg = ENCODER_CONFIGS[version]
    model = MAEModel(
        in_channels=HALF_CH,
        num_patches=cfg['num_patches'], patch_rows=cfg['patch_rows'],
        encoder_dim=EMBED_DIM, decoder_dim=96, num_heads=4,
        encoder_depth=4, use_rope=True,
        use_lane_type_embed=True, use_bpm_cond=False,
        lane_types=LANE_TYPES_HALF,
    )
    ckpt = Path(cfg['ckpt'])
    model.encoder.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.encoder.eval().to(DEVICE)
    print(f'encoder {version} loaded from {ckpt}')
    return model.encoder


@torch.no_grad()
def embed_chart_segments(encoder, npy_path: Path, temporal_diff: bool = False):
    """
    Returns
    -------
    fused      : (T, FUSE_DIM or FUSE_DIM_TD) float32
    bar_starts : (T,) int  — approximate bar index (chart start = bar 0)
    note_counts: (T,) int
    """
    enc, nc, _ = encode_chart_pretrain_v6(str(npy_path))
    p1_wins, p2_wins, win_nc = window_chart_time_half_v10(enc, nc)
    T = p1_wins.shape[0]
    if T == 0:
        return None, None, None

    x1  = torch.from_numpy(p1_wins[:, :, :HALF_CH]).float().to(DEVICE)
    x2  = torch.from_numpy(p2_wins[:, :, :HALF_CH]).float().to(DEVICE)
    bpm = torch.from_numpy(p1_wins[:, :, HALF_CH].mean(1)).float().to(DEVICE)

    if temporal_diff:
        p1 = encoder.embed_patches(x1, bpm=bpm).cpu().numpy()  # (T, P, 192)
        p2 = encoder.embed_patches(x2, bpm=bpm).cpu().numpy()
        m1, m2 = p1.mean(axis=1), p2.mean(axis=1)              # (T, 192) mean
        d1 = np.diff(p1, axis=1).mean(axis=1)                  # (T, 192) mean temporal diff
        d2 = np.diff(p2, axis=1).mean(axis=1)
        fused = np.concatenate([
            (m1 + m2) / 2.0, m1 - m2,   # symmetric/asymmetric mean
            (d1 + d2) / 2.0, d1 - d2,   # symmetric/asymmetric motion direction
        ], axis=1)                        # (T, 768)
    else:
        e1 = encoder.embed_segment(x1, bpm=bpm).cpu().numpy()  # (T, 192)
        e2 = encoder.embed_segment(x2, bpm=bpm).cpu().numpy()
        fused = np.concatenate([(e1 + e2) / 2.0, e1 - e2], axis=1)  # (T, 384)

    # approximate bar start from window index × stride
    bpm_arr = np.maximum(enc[:, 16] * BPM_SCALE, 1.0)
    avg_bpm = float(bpm_arr.mean())
    bars_per_sec = avg_bpm / (60.0 * 4)
    bar_starts = np.array(
        [int(i * STRIDE_SECS_V10 * bars_per_sec) for i in range(T)],
        dtype=np.int32,
    )
    return fused, bar_starts, win_nc


# ── Segment ASCII snapshot ────────────────────────────────────────────────────

def segment_ascii(npy_path: Path, bar_start: int, n_bars: int = 4) -> str:
    """Beat-grid ASCII: 1 line = 1 beat, 1 char = 1 lane. S=scratch X=key .=empty"""
    arr = np.load(npy_path)
    row_start = bar_start * ROWS_PER_BAR
    row_end   = min(row_start + n_bars * ROWS_PER_BAR, arr.shape[0])
    seg       = arr[row_start:row_end, :16]

    HEAD          = frozenset({1, 2, 4, 6, 8})
    rows_per_beat = ROWS_PER_BAR // 4
    n_beats       = (row_end - row_start) // rows_per_beat

    lines = []
    for b in range(n_beats):
        beat   = seg[b * rows_per_beat:(b + 1) * rows_per_beat]
        active = np.isin(beat, list(HEAD)).any(axis=0)
        chars  = [('S' if lane in (0, 15) else 'X') if active[lane] else '.'
                  for lane in range(16)]
        lines.append(chars[0] + ' ' + ''.join(chars[1:8]) + ' '
                     + ''.join(chars[8:15]) + ' ' + chars[15])
    return '\n'.join(lines)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--levels', type=int, nargs='+', default=[10, 11, 12])
    parser.add_argument('--top-reps', type=int, default=3)
    parser.add_argument('--nn-queries', type=int, default=5)
    parser.add_argument('--show-ascii', action='store_true')
    parser.add_argument('--top-dense', type=int, default=0,
                        help='if >0, only cluster the N densest segments per chart')
    parser.add_argument('--signatures', type=int, default=0,
                        help='if >0, show N signature segments per query chart')
    parser.add_argument('--rating', default='rating_hc',
                        choices=['rating_ec', 'rating_hc', 'rating_exh', 'rating_stat'],
                        help='which rating column to display')
    parser.add_argument('--encoder', default='v12', choices=list(ENCODER_CONFIGS),
                        help='which pretrained encoder to use')
    parser.add_argument('--query', nargs='+', default=[],
                        help='title substrings to use as NN queries (case-insensitive)')
    parser.add_argument('--query-frac', nargs=2, type=float, default=[0.0, 1.0],
                        metavar=('START', 'END'),
                        help='fractional position range [0,1] of the query chart to use '
                             'for its histogram (e.g. 0.75 1.0 = last quarter)')
    parser.add_argument('--query-bars', nargs='+', type=int, default=[],
                        metavar='BAR',
                        help='bar range [START] or [START END] to use for the query histogram')
    parser.add_argument('--show-bars', action='store_true',
                        help='for top NN match, show which bars share clusters with the query')
    parser.add_argument('--temporal-diff', action='store_true',
                        help='use mean(patch_diff) alongside mean(patches) to capture '
                             'sequential lane-motion patterns (e.g. staircases)')
    args = parser.parse_args()

    print(f'device: {DEVICE}', flush=True)
    encoder = load_encoder(args.encoder)
    fuse_dim = FUSE_DIM_TD if args.temporal_diff else FUSE_DIM
    print(f'encoder loaded  (fused embed_dim={fuse_dim}'
          f'{", +temporal-diff" if args.temporal_diff else ""})', flush=True)

    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest['status'] == 'ok'].reset_index(drop=True)

    # ── Embed all segments ────────────────────────────────────────────────────
    all_embs, all_meta = [], []
    npy_to_id, seg_chart_id_list = {}, []

    for level in args.levels:
        ldir = LEVEL_DIRS.get(level)
        if not ldir:
            continue
        subset = manifest[manifest['level'] == level]
        print(f'embedding lv{level}: {len(subset)} charts...', flush=True)

        for _, row in subset.iterrows():
            npy = DATA_ROOT / ldir / str(row['file_path'])
            if not npy.exists():
                continue
            fused, bar_starts, nc = embed_chart_segments(encoder, npy, args.temporal_diff)
            if fused is None:
                continue

            indices = np.arange(len(fused))
            if args.top_dense > 0 and len(indices) > args.top_dense:
                indices = np.argsort(nc)[::-1][:args.top_dense]

            key = str(npy)
            if key not in npy_to_id:
                npy_to_id[key] = len(npy_to_id)
            cid = npy_to_id[key]

            for i in indices:
                all_embs.append(fused[i])
                seg_chart_id_list.append(cid)
                all_meta.append({
                    'title':      str(row['title']),
                    'level':      int(level),
                    'npy':        key,
                    'bar_start':  int(bar_starts[i]),
                    'win_idx':    int(i),
                    'note_count': int(nc[i]),
                    'rating':     float(row[args.rating]) if pd.notna(row.get(args.rating)) else None,
                })

    X            = np.stack(all_embs).astype(np.float32)   # (N, 384)
    seg_chart_id = np.array(seg_chart_id_list, dtype=np.int32)
    print(f'\ntotal segments: {len(X):,}', flush=True)

    X_norm = normalize(X, norm='l2')

    # ── K-means ───────────────────────────────────────────────────────────────
    print(f'running K-means  (k={args.k})...', flush=True)
    km = MiniBatchKMeans(n_clusters=args.k, random_state=42,
                         batch_size=4096, n_init=10, max_iter=300)
    labels    = km.fit_predict(X_norm)
    centroids = km.cluster_centers_

    cluster_sizes = np.bincount(labels, minlength=args.k)
    print(f'cluster size range: {cluster_sizes.min()}–{cluster_sizes.max()}  '
          f'(mean {cluster_sizes.mean():.0f})', flush=True)

    # ── Representatives ───────────────────────────────────────────────────────
    print(f'\n{"="*70}')
    print('CLUSTER REPRESENTATIVES')
    print(f'{"="*70}')

    for k in range(args.k):
        idxs    = np.where(labels == k)[0]
        dists   = np.linalg.norm(X_norm[idxs] - centroids[k], axis=1)
        top_idx = idxs[np.argsort(dists)[:args.top_reps]]

        cmeta   = [all_meta[i] for i in idxs]
        lv_str  = '  '.join(f"lv{lv}:{sum(1 for m in cmeta if m['level']==lv)}"
                             for lv in args.levels)
        rated   = [m['rating'] for m in cmeta if m['rating'] is not None]
        rtag    = args.rating.split('_')[1]
        ec_str  = f"{rtag}={np.mean(rated):.1f}±{np.std(rated):.1f}" if rated else 'no rating'
        print(f'\nCluster {k:>2}  ({cluster_sizes[k]:>5} segs)  {lv_str}  {ec_str}')

        for rank, idx in enumerate(top_idx, 1):
            m   = all_meta[idx]
            ec  = f"ec={m['rating']:.1f}" if m['rating'] is not None else '      '
            print(f"  {rank}. [{m['level']}] {m['title'][:45]:<45}  bar~{m['bar_start']:>3}  "
                  f"nc={m['note_count']:>4}  {ec}")
            if args.show_ascii:
                art = segment_ascii(Path(m['npy']), m['bar_start'])
                for line in art.split('\n'):
                    print(f"      {line}")
                print()

    # ── Per-chart archetype histograms ────────────────────────────────────────
    # Also build side-swapped histograms: negate asymmetry component (last 192
    # dims) of each embedding — equivalent to swapping P1↔P2 sides — so that
    # NN retrieval is invariant to the left/right hand assignment of a chart.
    # Side-swap: negate the p1-p2 asymmetry components.
    # CLS mode:  dims [192:384] = p1-p2
    # TD mode:   dims [192:384] = mean(p1-p2), [576:768] = diff(p1-p2)
    X_swap = X_norm.copy()
    X_swap[:, EMBED_DIM:2*EMBED_DIM] *= -1
    if args.temporal_diff:
        X_swap[:, 3*EMBED_DIM:4*EMBED_DIM] *= -1
    swap_labels = km.predict(X_swap)

    chart_hist       = {}   # npy_path → (k,) float  (natural orientation)
    chart_hist_swap  = {}   # npy_path → (k,) float  (P1↔P2 swapped)
    chart_info       = {}
    chart_total_wins = {}   # npy_path → total window count (before top-dense filter)

    for i, meta in enumerate(all_meta):
        key = meta['npy']
        if key not in chart_hist:
            chart_hist[key]      = np.zeros(args.k, dtype=np.float32)
            chart_hist_swap[key] = np.zeros(args.k, dtype=np.float32)
            chart_info[key]      = meta
            chart_total_wins[key] = 0
        chart_hist[key][labels[i]]           += 1
        chart_hist_swap[key][swap_labels[i]] += 1
        if meta['win_idx'] >= chart_total_wins[key]:
            chart_total_wins[key] = meta['win_idx'] + 1

    for key in chart_hist:
        s = chart_hist[key].sum()
        if s > 0:
            chart_hist[key]      /= s
            chart_hist_swap[key] /= s

    keys_list        = list(chart_hist.keys())
    hist_matrix      = np.stack([chart_hist[k]      for k in keys_list])
    hist_matrix_swap = np.stack([chart_hist_swap[k] for k in keys_list])

    lv12_keys = [k for k in keys_list
                 if chart_info[k]['level'] == 12 and chart_info[k]['rating'] is not None]

    # ── Nearest-neighbour retrieval ───────────────────────────────────────────
    if args.nn_queries > 0 and lv12_keys:
        print(f'\n{"="*70}')
        print('NEAREST-NEIGHBOUR CHART RETRIEVAL  (Jensen-Shannon distance)')
        print(f'{"="*70}')

        if args.query:
            query_keys = []
            for pattern in args.query:
                pat_lower = pattern.lower()
                matched = [k for k in lv12_keys
                           if pat_lower in chart_info[k]['title'].lower()]
                if not matched:
                    matched = [k for k in keys_list
                               if pat_lower in chart_info[k]['title'].lower()]
                if matched:
                    query_keys.extend(matched)
                else:
                    print(f'  [warn] no chart matched "{pattern}"')
            query_keys = list(dict.fromkeys(query_keys))  # dedup, preserve order
        else:
            rng        = np.random.default_rng(42)
            query_keys = list(rng.choice(lv12_keys,
                                         size=min(args.nn_queries, len(lv12_keys)),
                                         replace=False))

        frac_start, frac_end = args.query_frac
        use_frac = (frac_start, frac_end) != (0.0, 1.0)
        bar_lo = args.query_bars[0] if len(args.query_bars) >= 1 else None
        bar_hi = args.query_bars[1] if len(args.query_bars) >= 2 else None
        use_bars = bar_lo is not None

        if use_bars:
            hi_str = f'–{bar_hi}' if bar_hi is not None else '+'
            print(f'  (query histogram restricted to bars {bar_lo}{hi_str})')
        elif use_frac:
            print(f'  (query histogram restricted to chart position '
                  f'{frac_start:.0%}–{frac_end:.0%})')

        for qk in query_keys:
            qi      = keys_list.index(qk)
            qm      = chart_info[qk]

            if use_bars:
                h_nat  = np.zeros(args.k, dtype=np.float32)
                h_swap = np.zeros(args.k, dtype=np.float32)
                for i, meta in enumerate(all_meta):
                    if meta['npy'] != qk:
                        continue
                    b = meta['bar_start']
                    if b < bar_lo:
                        continue
                    if bar_hi is not None and b >= bar_hi:
                        continue
                    h_nat[labels[i]]      += 1
                    h_swap[swap_labels[i]] += 1
                s = h_nat.sum()
                if s > 0:
                    h_nat /= s; h_swap /= s
                qh, qh_swap = h_nat, h_swap
            elif use_frac:
                T_q     = chart_total_wins[qk]
                lo, hi  = int(frac_start * T_q), int(frac_end * T_q)
                h_nat   = np.zeros(args.k, dtype=np.float32)
                h_swap  = np.zeros(args.k, dtype=np.float32)
                for i, meta in enumerate(all_meta):
                    if meta['npy'] == qk and lo <= meta['win_idx'] < hi:
                        h_nat[labels[i]]      += 1
                        h_swap[swap_labels[i]] += 1
                s = h_nat.sum()
                if s > 0:
                    h_nat /= s; h_swap /= s
                qh, qh_swap = h_nat, h_swap
            else:
                qh      = hist_matrix[qi]
                qh_swap = hist_matrix_swap[qi]

            d_nat  = np.array([jensenshannon(qh,      hist_matrix[j]) for j in range(len(keys_list))])
            d_swap = np.array([jensenshannon(qh_swap, hist_matrix[j]) for j in range(len(keys_list))])
            dists  = np.minimum(d_nat, d_swap)
            dists[qi] = 9999.0
            top5 = np.argsort(dists)[:6]

            ec_str = f"{args.rating.split('_')[1]}={qm['rating']:.1f}" if qm['rating'] else ''
            print(f"\nQuery: [{qm['level']}] {qm['title']}  {ec_str}")
            dom = qh.argmax()
            print(f"  dominant archetype: cluster {dom} ({100*qh[dom]:.0f}%)")
            rtag = args.rating.split('_')[1]
            print(f"  {'rank':>4}  {'dist':>6}  {'title':<45}  {'lv':>3}  {rtag:>6}")
            rank = 1
            top_match_j = None
            for j in top5:
                if j == qi:
                    continue
                m    = chart_info[keys_list[j]]
                ec   = f"{m['rating']:.1f}" if m['rating'] is not None else '  —  '
                tag  = ' ~' if d_swap[j] < d_nat[j] else '  '
                print(f"  {rank:>4}  {dists[j]:>6.4f}  {m['title']:<45}  {m['level']:>3}  {ec:>6}{tag}")
                if rank == 1:
                    top_match_j      = j
                    top_match_swapped = d_swap[j] < d_nat[j]
                rank += 1
                if rank > 5:
                    break

            if args.show_bars and top_match_j is not None:
                mk = keys_list[top_match_j]
                mm = chart_info[mk]
                # clusters present in query's dense section
                q_cluster_labels = top_match_swapped and swap_labels or labels
                q_segs   = [(all_meta[i]['win_idx'], labels[i]) for i, meta in enumerate(all_meta)
                            if meta['npy'] == qk]
                q_clusters = set(lbl for _, lbl in q_segs)
                # segments of the match chart that share those clusters
                use_lbls = swap_labels if top_match_swapped else labels
                m_segs = [(all_meta[i]['bar_start'], use_lbls[i], all_meta[i]['note_count'])
                          for i, meta in enumerate(all_meta)
                          if meta['npy'] == mk and use_lbls[i] in q_clusters]
                m_segs.sort()
                print(f"\n  Matching bars in {mm['title']}{'  (swapped)' if top_match_swapped else ''}:")
                print(f"  {'bar':>5}  {'cluster':>7}  {'nc':>5}")
                for bar, clust, nc in m_segs:
                    print(f"  {bar:>5}  {clust:>7}  {nc:>5}")

    # ── Signature segments ────────────────────────────────────────────────────
    if args.signatures > 0 and lv12_keys:
        print(f'\n{"="*70}')
        print('CHART SIGNATURE SEGMENTS  (highest centroid distance = most atypical)')
        print(f'{"="*70}')

        rng        = np.random.default_rng(42)
        sig_keys   = rng.choice(lv12_keys,
                                size=min(args.nn_queries, len(lv12_keys)),
                                replace=False)

        for qk in sig_keys:
            cid    = npy_to_id[qk]
            my_idx = np.where(seg_chart_id == cid)[0]
            if my_idx.size == 0:
                continue

            # distance to assigned centroid — high = unusual within its cluster
            centroid_dists = np.linalg.norm(
                X_norm[my_idx] - centroids[labels[my_idx]], axis=1
            )
            order   = np.argsort(centroid_dists)[::-1]
            top_sig = my_idx[order[:args.signatures]]

            qm     = chart_info[qk]
            ec_str = f"{args.rating.split('_')[1]}={qm['rating']:.1f}" if qm['rating'] else ''
            print(f"\n{qm['title']}  [{qm['level']}]  {ec_str}")
            print(f"  {'dist_to_ctr':>11}  {'bar':>4}  {'nc':>5}  {'cluster':>7}")
            for idx in top_sig:
                m = all_meta[idx]
                d = centroid_dists[np.where(my_idx == idx)[0][0]]
                print(f"  {d:>11.4f}  {m['bar_start']:>4}  {m['note_count']:>5}  "
                      f"{labels[idx]:>7}")
                if args.show_ascii:
                    art = segment_ascii(Path(m['npy']), m['bar_start'])
                    for line in art.split('\n'):
                        print(f"      {line}")
                    print()

    # ── Dominant archetype distribution — lv12 ────────────────────────────────
    print(f'\n{"="*70}')
    print('DOMINANT ARCHETYPE — lv12 charts')
    print(f'{"="*70}')
    dom_counts = np.zeros(args.k, dtype=int)
    for k in lv12_keys:
        dom_counts[chart_hist[k].argmax()] += 1
    for cid in np.argsort(dom_counts)[::-1][:12]:
        if dom_counts[cid] == 0:
            break
        # mean ec for charts dominated by this cluster
        ec_vals = [chart_info[k]['rating'] for k in lv12_keys
                   if chart_hist[k].argmax() == cid and chart_info[k]['rating']]
        ec_str = f"{args.rating.split('_')[1]}={np.mean(ec_vals):.1f}" if ec_vals else ''
        print(f"  cluster {cid:>2}: {dom_counts[cid]:>3} charts  {ec_str}")


if __name__ == '__main__':
    main()
