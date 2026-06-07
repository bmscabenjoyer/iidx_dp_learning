"""
Nearest-neighbour search over CausalEventEncoder window embeddings.

Augmented query search is always on: for each query window the encoder
produces 4 embeddings (original, side_swap, mirror, both). Each index
window is scored by the max similarity across all 4 variants, so
mirror-equivalent patterns are found regardless of orientation.

Modes
-----
  --query CHART_STEM                  best-matching window for this chart
  --query CHART_STEM --t-start N      window nearest N seconds
  --interactive                       loop: type chart names, get results
  --list                              list all indexed chart names

Usage
-----
    python search_event_index.py --query "IX_21__nine_ix_DAC00" --t-start 92
    python search_event_index.py --interactive
    python search_event_index.py --list
"""

import argparse
import sys
import numpy as np
from pathlib import Path


# ── Index ─────────────────────────────────────────────────────────────────────

def load_index(path):
    data = np.load(path, allow_pickle=True)
    idx = {
        'vectors':       data['vectors'],
        'chart_ids':     data['chart_ids'],
        'win_t_start':   data['win_t_start'],
        'chart_names':   data['chart_names'],
        'chart_levels':  data['chart_levels'],
    }
    if 'win_tok_start' in data:
        idx['win_tok_start'] = data['win_tok_start']
    return idx


# ── Encoder ───────────────────────────────────────────────────────────────────

def load_encoder(ckpt_path, device):
    import torch
    from causal_event_encoder import CausalEventEncoder
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    cfg  = ckpt['model_cfg']
    enc  = CausalEventEncoder(**cfg).to(device)
    enc.load_state_dict(ckpt['state_dict'])
    enc.eval()
    print(f'Encoder: epoch {ckpt["epoch"]}, loss={ckpt["loss"]:.4f}, d={cfg["d_model"]}')
    return enc


def _encode_one(encoder, lb, db, nt, device):
    import torch
    T    = len(db)
    lb_t = torch.from_numpy(lb).float().unsqueeze(0).to(device)
    db_t = torch.from_numpy(db).long().unsqueeze(0).to(device)
    nt_t = torch.from_numpy(nt).long().unsqueeze(0).to(device)
    pad  = torch.zeros(1, T, dtype=torch.bool, device=device)
    with torch.no_grad():
        return encoder.encode(lb_t, db_t, nt_t, pad).squeeze(0).cpu().numpy()


def encode_augmented(encoder, lm, nt, ts, device):
    """Encode a window with all 4 augmentations. Returns (4, d) float32."""
    from event_dataset   import masks_to_bits, _permute_masks, _SIDE_SWAP, _MIRROR
    from event_tokenize  import quantize_delta_vec

    T      = len(lm)
    deltas = np.zeros(T, dtype=np.float64)
    if T > 1:
        deltas[1:] = np.diff(ts.astype(np.float64))
    db = quantize_delta_vec(deltas).astype(np.int64)

    variants = [
        lm,
        _permute_masks(_permute_masks(lm, _SIDE_SWAP), _MIRROR),
    ]
    embs = [_encode_one(encoder, masks_to_bits(lm_v), db, nt.astype(np.int64), device)
            for lm_v in variants]
    return np.stack(embs)   # (4, d)


# ── Chart token loading ────────────────────────────────────────────────────────

def get_window_tokens(idx, global_wi, data_dir):
    """Return (lm, nt, ts) for the window at global index global_wi."""
    from event_tokenize       import chart_to_tokens
    from causal_event_dataset import WIN_TOKENS

    ci         = int(idx['chart_ids'][global_wi])
    name       = idx['chart_names'][ci]
    level      = int(idx['chart_levels'][ci])
    tok_start  = int(idx['win_tok_start'][global_wi])

    chart_path = Path(data_dir) / f'dp{level}_active/charts/{name}.npy'
    toks       = chart_to_tokens(chart_path)

    lm = toks['lane_masks'][tok_start : tok_start + WIN_TOKENS]
    nt = toks['note_types'] [tok_start : tok_start + WIN_TOKENS]
    ts = toks['times_sec']  [tok_start : tok_start + WIN_TOKENS].astype(np.float64)
    return lm, nt, ts


# ── Search ────────────────────────────────────────────────────────────────────

def search(idx, q_vecs, exclude_ci=None, top_charts=10, only_levels=None):
    """
    q_vecs : (d,) single vector  OR  (K, d) multiple variants → max sim used.
    Returns list of result dicts sorted by sim desc.
    """
    vecs = idx['vectors']   # (N, d)
    cids = idx['chart_ids']

    if q_vecs.ndim == 1:
        sims = vecs @ q_vecs
    else:
        sims = (vecs @ q_vecs.T).max(axis=1)   # (N,)

    if exclude_ci is not None:
        sims[cids == exclude_ci] = -2.0

    if only_levels:
        level_set = set(only_levels)
        for wi, ci in enumerate(cids):
            if int(idx['chart_levels'][ci]) not in level_set:
                sims[wi] = -2.0

    chart_best = {}
    for wi, (ci, s) in enumerate(zip(cids, sims)):
        if s <= -2.0:
            continue
        if ci not in chart_best or s > chart_best[ci][1]:
            chart_best[ci] = (wi, s)

    ranked = sorted(chart_best.items(), key=lambda x: -x[1][1])[:top_charts]
    return [
        {
            'chart_i': ci,
            'name':    idx['chart_names'][ci],
            'level':   int(idx['chart_levels'][ci]),
            'sim':     float(s),
            'win_i':   wi,
            't_start': float(idx['win_t_start'][wi]),
        }
        for ci, (wi, s) in ranked
    ]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_chart(idx, query):
    names = idx['chart_names']
    q = query.lower()
    matches = [i for i, n in enumerate(names) if q in n.lower()]
    if not matches:
        return None, []
    if len(matches) == 1:
        return matches[0], []
    exact = [i for i in matches if names[i].lower() == q]
    if exact:
        return exact[0], []
    return None, matches


def window_at_time(idx, chart_i, t_sec):
    """Global index of the window in chart_i whose t_start is closest to t_sec."""
    mask = np.where(idx['chart_ids'] == chart_i)[0]
    ts   = idx['win_t_start'][mask]
    return int(mask[np.argmin(np.abs(ts - t_sec))])


def fmt_time(sec):
    m, s = divmod(int(sec), 60)
    return f'{m}:{s:02d}'


def print_results(results, query_name, query_level, t_start=None, augmented=True):
    aug_str = ' [+flip]' if augmented else ''
    t_str   = f' t={fmt_time(t_start)}' if t_start is not None else ' (chart mean)'
    print(f'\nQuery: {query_name} (lv{query_level}){t_str}{aug_str}')
    print(f'{"Rank":<5} {"Sim":>6}  {"Lv":>3}  {"t":>7}  Name')
    print('─' * 72)
    for rank, r in enumerate(results, 1):
        print(f'{rank:<5} {r["sim"]:>6.3f}  lv{r["level"]}  '
              f'{fmt_time(r["t_start"]):>7}  {r["name"]}')


def _query_and_print(idx, encoder, ci, global_wi, data_dir, top, only_levels):
    t_start = float(idx['win_t_start'][global_wi])
    if encoder is not None and 'win_tok_start' in idx:
        lm, nt, ts = get_window_tokens(idx, global_wi, data_dir)
        q_vecs     = encode_augmented(encoder, lm, nt, ts,
                                      next(encoder.parameters()).device)
    else:
        q_vecs = idx['vectors'][global_wi]

    results = search(idx, q_vecs, exclude_ci=ci, top_charts=top,
                     only_levels=only_levels)
    print_results(results, idx['chart_names'][ci], int(idx['chart_levels'][ci]),
                  t_start=t_start, augmented=(encoder is not None))


# ── Interactive loop ──────────────────────────────────────────────────────────

def interactive_loop(idx, encoder, data_dir, only_levels, top):
    print('Interactive search. Type chart name (partial ok).')
    print('Append ":Ns" for window nearest N seconds. Example: "quasar:27s"')
    if only_levels:
        print(f'Filtering to: lv{only_levels}')
    print()

    while True:
        try:
            raw = input('query> ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not raw or raw.lower() in ('quit', 'exit', 'q'):
            break

        t_query = None
        if ':' in raw:
            stem, w = raw.rsplit(':', 1)
            w = w.strip()
            try:
                t_query = float(w[:-1]) if w.endswith('s') else None
                raw = stem.strip()
            except ValueError:
                pass

        ci, candidates = find_chart(idx, raw)
        if ci is None:
            if candidates:
                print(f'Ambiguous ({len(candidates)} matches):')
                for i in candidates[:10]:
                    print(f'  lv{idx["chart_levels"][i]}  {idx["chart_names"][i]}')
            else:
                print('No match found.')
            continue

        if t_query is not None:
            global_wi = window_at_time(idx, ci, t_query)
        else:
            # use window with highest mean similarity to rest of chart
            q_mask    = np.where(idx['chart_ids'] == ci)[0]
            mean_sim  = (idx['vectors'][q_mask] @ idx['vectors'].T).mean(axis=1)
            global_wi = int(q_mask[np.argmax(mean_sim)])

        _query_and_print(idx, encoder, ci, global_wi, data_dir, top, only_levels)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--index',    default='embeddings/causal_index_v2.npz')
    ap.add_argument('--ckpt',     default='checkpoints/causal_pretrain_v2/encoder_best.pt')
    ap.add_argument('--data-dir', default=str(Path.home() / 'projects/iidx_data'))
    ap.add_argument('--query',    default=None)
    ap.add_argument('--t-start',  type=float, default=None)
    ap.add_argument('--top',      type=int,   default=10)
    ap.add_argument('--levels',   nargs='+',  type=int, default=None)
    ap.add_argument('--no-augment', action='store_true',
                    help='skip flip augmentation (faster, less accurate)')
    ap.add_argument('--interactive', action='store_true')
    ap.add_argument('--list',        action='store_true')
    args = ap.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    idx = load_index(args.index)
    print(f'Index: {len(idx["chart_names"])} charts, '
          f'{len(idx["vectors"])} windows, d={idx["vectors"].shape[1]}')

    encoder = None
    if not args.no_augment and 'win_tok_start' in idx:
        ckpt_p = Path(args.ckpt)
        if ckpt_p.exists():
            encoder = load_encoder(str(ckpt_p), device)
        else:
            print(f'Checkpoint not found ({ckpt_p}), falling back to no augmentation.')

    if args.list:
        for i, (n, l) in enumerate(zip(idx['chart_names'], idx['chart_levels'])):
            print(f'lv{l}  {n}')
        return

    if args.interactive:
        interactive_loop(idx, encoder, args.data_dir, args.levels, args.top)
        return

    if args.query:
        ci, candidates = find_chart(idx, args.query)
        if ci is None:
            if candidates:
                print(f'Ambiguous ({len(candidates)} matches):')
                for i in candidates[:20]:
                    print(f'  lv{idx["chart_levels"][i]}  {idx["chart_names"][i]}')
            else:
                print('No match.')
            sys.exit(1)

        if args.t_start is not None:
            global_wi = window_at_time(idx, ci, args.t_start)
        else:
            q_mask   = np.where(idx['chart_ids'] == ci)[0]
            mean_sim = (idx['vectors'][q_mask] @ idx['vectors'].T).mean(axis=1)
            global_wi = int(q_mask[np.argmax(mean_sim)])

        _query_and_print(idx, encoder, ci, global_wi, args.data_dir,
                         args.top, args.levels)
        return

    ap.print_help()


if __name__ == '__main__':
    main()
