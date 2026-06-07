"""
Build a nearest-neighbour search index from CausalEventEncoder embeddings.

Usage
-----
    python build_causal_index.py                        # lv10/11/12
    python build_causal_index.py --levels 12            # lv12 only
    python build_causal_index.py --ckpt checkpoints/causal_pretrain_v2/encoder_best.pt
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from event_tokenize       import chart_to_tokens, quantize_delta_vec
from event_dataset        import masks_to_bits
from causal_event_dataset import WIN_TOKENS, STRIDE_TOKENS, MAX_TOKENS
from causal_event_encoder import CausalEventEncoder


def windows_from_tokens(toks, win_tokens=WIN_TOKENS, stride_tokens=STRIDE_TOKENS):
    """Yield (lane_bits, delta_bins, note_types, is_padded, tok_start, t_start) per window.

    Slides by token index. Every window has exactly win_tokens real tokens, no padding.
    tok_start is the token index of the first token in the window.
    t_start is the physical time (seconds) of that token, for display.
    """
    lm = toks['lane_masks']
    nt = toks['note_types']
    ts = toks['times_sec'].astype(np.float64)
    n  = len(ts)

    if n < win_tokens:
        return

    t0_idx = 0
    while t0_idx + win_tokens <= n:
        idx = np.arange(t0_idx, t0_idx + win_tokens)
        T   = win_tokens

        deltas        = np.zeros(T, dtype=np.float64)
        deltas[1:]    = np.diff(ts[idx])
        db            = quantize_delta_vec(deltas).astype(np.int64)
        lb            = masks_to_bits(lm[idx])
        nt_w          = nt[idx].astype(np.int64)
        is_padded     = np.zeros(T, dtype=bool)  # never any padding

        yield lb, db, nt_w, is_padded, t0_idx, float(ts[t0_idx])
        t0_idx += stride_tokens


@torch.no_grad()
def embed_batch(encoder, lb, db, nt, pad, device, pool='last'):
    lb  = torch.from_numpy(lb).float().to(device)
    db  = torch.from_numpy(db).long().to(device)
    nt  = torch.from_numpy(nt).long().to(device)
    pad = torch.from_numpy(pad).bool().to(device)
    return encoder.encode(lb, db, nt, pad, pool=pool).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',          default='checkpoints/causal_pretrain_v2/encoder_best.pt')
    ap.add_argument('--data-dir',      default=str(Path.home() / 'projects/iidx_data'))
    ap.add_argument('--out',           default='embeddings/causal_index_v2.npz')
    ap.add_argument('--levels',        nargs='+', type=int, default=[10, 11, 12])
    ap.add_argument('--batch-size',    type=int, default=512)
    ap.add_argument('--pool',          default='last', choices=['last', 'mean'])
    ap.add_argument('--win-tokens',    type=int, default=WIN_TOKENS)
    ap.add_argument('--stride-tokens', type=int, default=STRIDE_TOKENS)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg  = ckpt['model_cfg']
    encoder = CausalEventEncoder(**cfg).to(device)
    encoder.load_state_dict(ckpt['state_dict'])
    encoder.eval()
    print(f'Loaded encoder (epoch {ckpt["epoch"]}, loss={ckpt["loss"]:.4f}, '
          f'd_model={cfg["d_model"]}, pool={args.pool})')

    data_dir = Path(args.data_dir)
    charts = []
    for level in args.levels:
        for p in sorted((data_dir / f'dp{level}_active/charts').glob('*.npy')):
            charts.append((level, p))
    print(f'Charts to embed: {len(charts)} (levels {args.levels})')

    all_vecs      = []
    all_chart_i   = []
    all_tok_start = []
    all_t_start   = []
    chart_names   = []
    chart_levels  = []

    for chart_idx, (level, path) in enumerate(charts):
        if chart_idx % 100 == 0:
            print(f'  {chart_idx}/{len(charts)} ...', flush=True)
        try:
            toks = chart_to_tokens(path)
        except Exception as e:
            print(f'  ERR {path.name}: {e}')
            continue

        rows = list(windows_from_tokens(
            toks,
            win_tokens    = args.win_tokens,
            stride_tokens = args.stride_tokens,
        ))
        if not rows:
            continue

        chart_names.append(path.stem)
        chart_levels.append(level)
        ci = len(chart_names) - 1

        for batch_start in range(0, len(rows), args.batch_size):
            batch = rows[batch_start:batch_start + args.batch_size]
            vecs = embed_batch(
                encoder,
                np.stack([r[0] for r in batch]),
                np.stack([r[1] for r in batch]),
                np.stack([r[2] for r in batch]),
                np.stack([r[3] for r in batch]),
                device,
                pool=args.pool,
            )
            all_vecs.append(vecs)
            all_chart_i.extend([ci] * len(batch))
            all_tok_start.extend(r[4] for r in batch)
            all_t_start.extend(r[5] for r in batch)

    print(f'Done. {len(chart_names)} charts, {len(all_chart_i)} windows total.')

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vectors      = np.concatenate(all_vecs, axis=0).astype(np.float32),
        chart_ids    = np.array(all_chart_i,    dtype=np.int32),
        win_tok_start= np.array(all_tok_start,  dtype=np.int32),
        win_t_start  = np.array(all_t_start,    dtype=np.float32),
        chart_names  = np.array(chart_names),
        chart_levels = np.array(chart_levels,   dtype=np.int8),
    )
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
