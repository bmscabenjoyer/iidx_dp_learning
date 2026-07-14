"""Fast weight optimization for simulator.py.

Precomputes per-chart feature averages (one pass), then optimises weights
by maximising Spearman ρ on EC/HC/EXH ratings.  Runs in seconds.

Approximation:  θ* ≈ d_avg + δ(n_total)
where d_avg is linear in weights given precomputed features.
This is accurate enough for ranking (the monotone relationship holds).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from scipy.optimize import differential_evolution
import sys; sys.path.insert(0, '.')
from simulator import (
    extract_events, chord_features, position_of,
    TENDON_PAIRS, MEDIUM_PAIRS, ROWS_PER_BEAT,
    ACTION_VALUES, P1_LANES, P2_LANES,
)

DATA_ROOT = Path('/home/jysuh/projects/iidx_data')

# ── per-chart feature extraction ──────────────────────────────────────────────

def extract_chart_features(events, total_notes, bpm_ref=150.0, bpm_exp=0.7):
    """Return per-chart feature averages for the linear d_avg model."""
    if not events or total_notes == 0:
        return None

    feats_n, feats_conflict, feats_adj, feats_scr = [], [], [], []
    feats_switch, feats_reach = [], []
    feats_tendon_trill, feats_medium_trill = [], []
    feats_wide_tp = []

    ref_nps = bpm_ref / 60.0 * 4

    prev_p1, prev_p2 = (), ()
    prev_row_p1 = prev_row_p2 = 0

    for row, p1, p2, bpm, s1, s2 in events:
        int_p1 = row - prev_row_p1 if p1 else ROWS_PER_BEAT
        int_p2 = row - prev_row_p2 if p2 else ROWS_PER_BEAT

        for keys, prev_keys, interval in [(p1, prev_p1, int_p1), (p2, prev_p2, int_p2)]:
            if not keys:
                continue
            n = len(keys)
            interval_secs = max(interval, 12) / (bpm / 60.0 * ROWS_PER_BEAT)
            nps = 1.0 / interval_secs
            tp = (nps / ref_nps) ** bpm_exp

            if n >= 5:
                feats_wide_tp.append(tp)
                continue

            conflict, n_adj, scr_side, _, _ = chord_features(keys)

            tendon_trill = medium_trill = 0
            pos_switch = center_jump = 0
            if prev_keys:
                pos_switch = int(
                    position_of(prev_keys) != position_of(keys)
                    and 'neutral' not in [position_of(prev_keys), position_of(keys)]
                )
                center_jump = abs(sum(keys)/n - sum(prev_keys)/len(prev_keys))
                prev_ks, cur_ks = set(prev_keys), set(keys)
                tendon_trill = sum(
                    1 for pair in TENDON_PAIRS
                    if (pair & prev_ks) and (pair & cur_ks)
                    and (pair & prev_ks) != (pair & cur_ks)
                )
                medium_trill = sum(
                    1 for pair in MEDIUM_PAIRS
                    if (pair & prev_ks) and (pair & cur_ks)
                    and (pair & prev_ks) != (pair & cur_ks)
                )

            feats_n.append(n * tp)
            feats_conflict.append(conflict * tp)
            feats_adj.append(n_adj * tp)
            feats_scr.append(scr_side * tp)
            feats_switch.append(pos_switch * tp)
            feats_reach.append(center_jump * tp)
            feats_tendon_trill.append(tendon_trill * tp)
            feats_medium_trill.append(medium_trill * tp)

        if p1: prev_p1, prev_row_p1 = p1, row
        if p2: prev_p2, prev_row_p2 = p2, row

    def safe_mean(lst, default=0.0):
        return float(np.mean(lst)) if lst else default

    if len(events) >= 2:
        avg_bpm = float(np.mean([e[3] for e in events]))
        first, last = events[0][0], events[-1][0]
        chart_secs = max((last - first) / (avg_bpm / 60.0 * ROWS_PER_BEAT), 1.0)
        chart_nps = total_notes / chart_secs
    else:
        chart_nps = ref_nps

    return {
        'fn'      : safe_mean(feats_n),
        'fconfl'  : safe_mean(feats_conflict),
        'fadj'    : safe_mean(feats_adj),
        'fscr'    : safe_mean(feats_scr),
        'fswitch' : safe_mean(feats_switch),
        'freach'  : safe_mean(feats_reach),
        'ftendon' : safe_mean(feats_tendon_trill),
        'fmedium' : safe_mean(feats_medium_trill),
        'fwide'   : safe_mean(feats_wide_tp),
        'chart_nps': chart_nps,
        'total_notes': total_notes,
    }


def d_avg_from_weights(f, w):
    """Predicted average difficulty from precomputed features and weight vector."""
    ref_nps = w[10] / 60.0 * 4   # w[10]=bpm_ref
    density_offset = min(w[11] * max(0.0, f['chart_nps'] - ref_nps), w[12])
    return (w[0]                  # bias
          + w[1]  * f['fn']
          + w[2]  * f['fconfl']
          + w[3]  * f['fadj']
          + w[4]  * f['fscr']
          + w[5]  * f['fswitch']
          + w[6]  * f['freach']
          + w[7]  * f['ftendon']
          + w[8]  * f['fmedium']
          + w[9]  * f['fwide']
          + density_offset)


def neg_rho_ec(w, chart_feats, targets):
    preds = np.array([d_avg_from_weights(f, w) for f in chart_feats])
    return -spearmanr(preds, targets).statistic


def main():
    manifest = pd.read_csv(DATA_ROOT / 'labeled_manifest.csv')
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    DP12_DIR = DATA_ROOT / 'dp12_active'
    print("Extracting chart features...")
    chart_feats, targets_ec, targets_hc, targets_exh = [], [], [], []
    for _, row in lv12.iterrows():
        npy = DP12_DIR / row['file_path']
        if not npy.exists():
            continue
        chart = np.load(npy)
        events = extract_events(chart)
        total = sum(len(p1)+len(p2)+s1+s2 for _,p1,p2,_,s1,s2 in events)
        f = extract_chart_features(events, total)
        if f is None:
            continue
        chart_feats.append(f)
        targets_ec.append(float(row['rating_ec']))
        targets_hc.append(float(row['rating_hc']))
        targets_exh.append(float(row['rating_exh']))

    targets_ec = np.array(targets_ec)
    print(f"  {len(chart_feats)} charts")

    # Weight vector:
    # [bias, w_n, w_conflict, w_adj, w_scr, w_switch, w_reach, w_tendon, w_medium, w_wide, bpm_ref, w_density, w_density_cap]
    w0 = [0.5, 0.6, 2.0, 0.8, 0.4, 0.6, 0.2, 1.5, 0.4, 0.5, 150.0, 0.5, 3.5]

    print(f"Initial EC ρ = {-neg_rho_ec(w0, chart_feats, targets_ec):.4f}")

    bounds = [
        (0.0, 5.0),   # bias
        (0.0, 3.0),   # w_n
        (0.0, 5.0),   # w_conflict
        (0.0, 3.0),   # w_adj
        (0.0, 3.0),   # w_scr
        (0.0, 3.0),   # w_switch
        (0.0, 2.0),   # w_reach
        (0.0, 5.0),   # w_tendon
        (0.0, 3.0),   # w_medium
        (0.0, 3.0),   # w_wide
        (100.0, 200.0),  # bpm_ref
        (0.0, 3.0),   # w_density
        (0.0, 10.0),  # w_density_cap
    ]

    print("Optimising (differential evolution, ~60s)...")
    result = differential_evolution(
        neg_rho_ec, bounds,
        args=(chart_feats, targets_ec),
        maxiter=200, popsize=12, seed=42,
        tol=1e-4, workers=1,
        callback=lambda xk, convergence: print(f"  ρ={-neg_rho_ec(xk, chart_feats, targets_ec):.4f}", flush=True) if np.random.random() < 0.03 else None,
    )

    w_opt = result.x
    print(f"\nOptimised EC ρ = {-result.fun:.4f}")
    names = ['bias', 'w_n', 'w_conflict', 'w_adj', 'w_scr', 'w_switch', 'w_reach',
             'w_tendon', 'w_medium', 'w_wide', 'bpm_ref', 'w_density', 'w_density_cap']
    print("Optimal weights:")
    for name, val in zip(names, w_opt):
        print(f"  {name:<15} = {val:.4f}")

    # also check HC/EXH ρ with these weights
    for tgt, name in [(targets_hc, 'hc'), (targets_exh, 'exh')]:
        preds = np.array([d_avg_from_weights(f, w_opt) for f in chart_feats])
        print(f"  {name} ρ = {spearmanr(preds, tgt).statistic:.4f}")


if __name__ == '__main__':
    main()
