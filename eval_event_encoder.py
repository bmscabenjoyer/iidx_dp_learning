"""
Discriminability evaluation for the EventEncoder CLS embedding.

Tests
-----
1. Within-chart vs cross-chart cosine similarity
   Good encoder: intra >> inter (same chart windows should cluster)

2. Level linear probe (lv10 / lv11 / lv12 — 3-class)
   Good encoder: accuracy well above 33% chance baseline

3. Nearest-neighbour retrieval sanity
   Prints the top-5 nearest windows for a few query charts

Usage
-----
    python eval_event_encoder.py
    python eval_event_encoder.py --ckpt checkpoints/event_pretrain/encoder_best.pt
    python eval_event_encoder.py --charts-per-level 50
"""

import argparse
import random
import numpy as np
import torch
from pathlib import Path

from event_tokenize import chart_to_tokens, quantize_delta_vec
from event_dataset  import masks_to_bits, WIN_SEC, STRIDE_SEC, MAX_TOKENS, MIN_TOKENS
from event_encoder  import EventEncoder

# ── helpers ───────────────────────────────────────────────────────────────────

def windows_from_tokens(toks: dict, win_sec=WIN_SEC, stride_sec=STRIDE_SEC,
                         max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS):
    """Yield (lane_bits, delta_bins, note_types, is_padded) arrays per window."""
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
            yield lb, db, nt_pad, is_padded
        t0 += stride_sec


@torch.no_grad()
def embed_chart(encoder: EventEncoder, chart_path: Path, device) -> np.ndarray:
    """Returns (N_windows, d_model) embeddings for one chart."""
    try:
        toks = chart_to_tokens(chart_path)
    except Exception:
        return np.empty((0,), dtype=np.float32)
    if len(toks['times_sec']) < MIN_TOKENS:
        return np.empty((0,), dtype=np.float32)

    rows = list(windows_from_tokens(toks))
    if not rows:
        return np.empty((0,), dtype=np.float32)

    lb  = torch.from_numpy(np.stack([r[0] for r in rows])).float().to(device)
    db  = torch.from_numpy(np.stack([r[1] for r in rows])).long().to(device)
    nt  = torch.from_numpy(np.stack([r[2] for r in rows])).long().to(device)
    pad = torch.from_numpy(np.stack([r[3] for r in rows])).bool().to(device)

    emb = encoder.encode(lb, db, nt, pad)           # (N, d)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().numpy()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='checkpoints/event_pretrain/encoder_best.pt')
    ap.add_argument('--data-dir', default=str(Path.home() / 'projects/iidx_data'))
    ap.add_argument('--charts-per-level', type=int, default=40)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ── Load encoder ──────────────────────────────────────────────────────────

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=True)
    cfg  = ckpt['model_cfg']
    encoder = EventEncoder(**cfg).to(device)
    encoder.load_state_dict(ckpt['state_dict'])
    encoder.eval()
    print(f'Loaded encoder (epoch {ckpt["epoch"]}, loss {ckpt["loss"]:.4f})')

    # ── Sample charts ─────────────────────────────────────────────────────────

    data_dir = Path(args.data_dir)
    all_charts = []   # list of (level, path)
    for level in [10, 11, 12]:
        paths = sorted((data_dir / f'dp{level}_active/charts').glob('*.npy'))
        sampled = random.sample(paths, min(args.charts_per_level, len(paths)))
        all_charts.extend((level, p) for p in sampled)

    print(f'Embedding {len(all_charts)} charts ...')

    # ── Embed ─────────────────────────────────────────────────────────────────

    chart_embs  = []   # (N_windows, d) per chart
    chart_levels = []
    chart_names  = []

    for level, path in all_charts:
        emb = embed_chart(encoder, path, device)
        if emb.ndim == 1:   # empty
            continue
        chart_embs.append(emb)
        chart_levels.append(level)
        chart_names.append(path.stem)

    print(f'  {len(chart_embs)} charts with embeddings, '
          f'{sum(len(e) for e in chart_embs)} total windows')

    # ── Test 1: within-chart vs cross-chart similarity ────────────────────────

    print('\n── Test 1: Intra vs Inter cosine similarity ──────────────────────')

    intra_sims, inter_sims = [], []
    rng = np.random.default_rng(args.seed)

    for i, emb in enumerate(chart_embs):
        if len(emb) < 2:
            continue
        # intra: random pairs within same chart
        idx = rng.choice(len(emb), size=(min(50, len(emb)), 2), replace=True)
        sims = (emb[idx[:, 0]] * emb[idx[:, 1]]).sum(-1)
        intra_sims.extend(sims[idx[:, 0] != idx[:, 1]].tolist())

        # inter: random pairs from a different chart
        j = rng.integers(len(chart_embs))
        while j == i or len(chart_embs[j]) == 0:
            j = rng.integers(len(chart_embs))
        n = min(50, len(emb), len(chart_embs[j]))
        a = emb[rng.choice(len(emb),           size=n, replace=True)]
        b = chart_embs[j][rng.choice(len(chart_embs[j]), size=n, replace=True)]
        inter_sims.extend((a * b).sum(-1).tolist())

    intra = np.array(intra_sims)
    inter = np.array(inter_sims)
    print(f'  Intra-chart  cosine: mean={intra.mean():.3f}  std={intra.std():.3f}')
    print(f'  Inter-chart  cosine: mean={inter.mean():.3f}  std={inter.std():.3f}')
    print(f'  Gap (intra-inter):   {intra.mean()-inter.mean():.3f}')

    # ── Test 2: level linear probe ────────────────────────────────────────────

    print('\n── Test 2: Level linear probe (lv10 / lv11 / lv12) ──────────────')
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score

    # mean-pool windows per chart → one vector per chart
    X = np.stack([e.mean(0) for e in chart_embs])
    y = np.array(chart_levels)

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    clf   = LogisticRegression(max_iter=1000, C=1.0)
    accs  = cross_val_score(clf, X_sc, y, cv=5, scoring='accuracy')
    print(f'  5-fold CV accuracy: {accs.mean():.3f} ± {accs.std():.3f}')
    print(f'  Chance baseline:    0.333')

    # ── Test 3: nearest-neighbour retrieval ───────────────────────────────────

    print('\n── Test 3: Nearest-neighbour retrieval (5 queries) ──────────────')

    # Build flat index: (total_windows, d)
    all_vecs   = np.concatenate(chart_embs, axis=0)             # (M, d)
    win_chart  = np.concatenate([
        np.full(len(e), i) for i, e in enumerate(chart_embs)
    ])                                                           # (M,)

    query_charts = random.sample(range(len(chart_embs)), min(5, len(chart_embs)))
    for qi in query_charts:
        q_vec  = chart_embs[qi].mean(0, keepdims=True)          # (1, d)
        sims   = (all_vecs * q_vec).sum(-1)                     # (M,)
        top5   = np.argsort(sims)[::-1][:6]
        print(f'\n  Query: {chart_names[qi]} (lv{chart_levels[qi]})')
        seen = set()
        for k in top5:
            ci = win_chart[k]
            if ci in seen:
                continue
            seen.add(ci)
            marker = '<-- query' if ci == qi else ''
            print(f'    sim={sims[k]:.3f}  lv{chart_levels[ci]}  '
                  f'{chart_names[ci]}  {marker}')
            if len(seen) >= 5:
                break


if __name__ == '__main__':
    main()
