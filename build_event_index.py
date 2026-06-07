"""
Build a nearest-neighbour search index from EventEncoder embeddings.

Embeds every window of every dp10/11/12 chart and saves per-window vectors
to embeddings/event_index.npz.

Usage
-----
    python build_event_index.py
    python build_event_index.py --ckpt checkpoints/event_pretrain/encoder_best.pt
    python build_event_index.py --levels 12          # only lv12
    python build_event_index.py --batch-size 512
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from event_tokenize import chart_to_tokens, quantize_delta_vec
from event_dataset  import masks_to_bits, WIN_SEC, STRIDE_SEC, MAX_TOKENS, MIN_TOKENS
from event_encoder  import EventEncoder


def windows_from_tokens(toks, win_sec=WIN_SEC, stride_sec=STRIDE_SEC,
                         max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS):
    """Yield (lane_bits, delta_bins, note_types, is_padded, t_start) per window."""
    lm = toks['lane_masks']
    nt = toks['note_types']
    ts = toks['times_sec'].astype(np.float64)
    t_max = float(ts[-1]) if len(ts) else 0.0
    t0 = 0.0
    while t0 < t_max:
        idx = np.where((ts >= t0) & (ts < t0 + win_sec))[0]
        if len(idx) >= min_tokens:
            idx = idx[:max_tokens]
            T   = len(idx)
            deltas = np.zeros(T, dtype=np.float64)
            if T > 1:
                deltas[1:] = np.diff(ts[idx])
            db  = quantize_delta_vec(deltas).astype(np.int64)
            lb  = masks_to_bits(lm[idx])
            pad = max_tokens - T
            if pad > 0:
                lb = np.pad(lb, ((0, pad), (0, 0)))
                db = np.pad(db, (0, pad))
                nt_pad = np.pad(nt[idx].astype(np.int64), (0, pad))
            else:
                nt_pad = nt[idx].astype(np.int64)
            is_padded = np.zeros(max_tokens, dtype=bool)
            is_padded[T:] = True
            yield lb, db, nt_pad, is_padded, float(t0)
        t0 += stride_sec


@torch.no_grad()
def embed_batch(encoder, lb, db, nt, pad, device):
    lb  = torch.from_numpy(lb).float().to(device)
    db  = torch.from_numpy(db).long().to(device)
    nt  = torch.from_numpy(nt).long().to(device)
    pad = torch.from_numpy(pad).bool().to(device)
    emb = encoder.encode(lb, db, nt, pad)
    return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt',       default='checkpoints/event_pretrain/encoder_best.pt')
    ap.add_argument('--data-dir',   default=str(Path.home() / 'projects/iidx_data'))
    ap.add_argument('--out',        default='embeddings/event_index.npz')
    ap.add_argument('--levels',     nargs='+', type=int, default=[10, 11, 12])
    ap.add_argument('--batch-size', type=int, default=512)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load encoder ──────────────────────────────────────────────────────────

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg  = ckpt['model_cfg']
    encoder = EventEncoder(**cfg).to(device)
    encoder.load_state_dict(ckpt['state_dict'])
    encoder.eval()
    print(f'Loaded encoder (epoch {ckpt["epoch"]}, d_model={cfg["d_model"]})')

    # ── Collect charts ────────────────────────────────────────────────────────

    data_dir = Path(args.data_dir)
    charts = []
    for level in args.levels:
        for p in sorted((data_dir / f'dp{level}_active/charts').glob('*.npy')):
            charts.append((level, p))
    print(f'Charts to embed: {len(charts)}')

    # ── Embed ─────────────────────────────────────────────────────────────────

    all_vecs    = []
    all_chart_i = []
    all_t_start = []
    chart_names = []
    chart_levels = []

    for chart_idx, (level, path) in enumerate(charts):
        if chart_idx % 200 == 0:
            print(f'  {chart_idx}/{len(charts)} ...', flush=True)
        try:
            toks = chart_to_tokens(path)
        except Exception as e:
            print(f'  ERR {path.name}: {e}')
            continue
        if len(toks['times_sec']) < MIN_TOKENS:
            continue

        rows = list(windows_from_tokens(toks))
        if not rows:
            continue

        chart_names.append(path.stem)
        chart_levels.append(level)
        ci = len(chart_names) - 1

        # Process in batches
        for batch_start in range(0, len(rows), args.batch_size):
            batch = rows[batch_start:batch_start + args.batch_size]
            lb  = np.stack([r[0] for r in batch])
            db  = np.stack([r[1] for r in batch])
            nt  = np.stack([r[2] for r in batch])
            pad = np.stack([r[3] for r in batch])
            ts  = [r[4] for r in batch]

            vecs = embed_batch(encoder, lb, db, nt, pad, device)
            all_vecs.append(vecs)
            all_chart_i.extend([ci] * len(batch))
            all_t_start.extend(ts)

    print(f'Done. {len(chart_names)} charts, {len(all_chart_i)} windows total.')

    # ── Save ──────────────────────────────────────────────────────────────────

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        vectors      = np.concatenate(all_vecs, axis=0).astype(np.float32),
        chart_ids    = np.array(all_chart_i, dtype=np.int32),
        win_t_start  = np.array(all_t_start,  dtype=np.float32),
        chart_names  = np.array(chart_names),
        chart_levels = np.array(chart_levels, dtype=np.int8),
    )
    print(f'Saved → {out}')


if __name__ == '__main__':
    main()
