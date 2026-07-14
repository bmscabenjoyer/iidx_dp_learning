"""
Rule-based IIDX DP difficulty simulator.

Per-note difficulty is computed analytically from chord ergonomics and
timing context — no encoder, no GPU.

Gauge simulation follows exact IIDX physics note-by-note, vectorised
over a grid of N_THETA hypothetical player skill levels θ.

Rating = θ* = minimum skill at which pass probability crosses 0.5.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
MANIFEST  = DATA_ROOT / 'labeled_manifest.csv'
DP12_DIR  = DATA_ROOT / 'dp12_active'

ACTION_VALUES = {1, 2, 4, 6, 8}     # tap, CN/HCN head, BSS/MSS head
P1_LANES      = list(range(1, 8))    # lanes 1–7  → positions 1–7
P2_LANES      = list(range(8, 15))   # lanes 8–14 → positions 1–7 (lane - 7)
ROWS_PER_BEAT = 48

# ── θ grid ────────────────────────────────────────────────────────────────────
THETA = np.linspace(0.0, 14.0, 200)

# ── Finger position framework ─────────────────────────────────────────────────
# Home positions (standard DP hand placement):
#   Pos A: Pinky=1  Ring=3  Middle=4  Index=5  Thumb=7
#   Pos B: Pinky=1  Ring=2  Middle=4  Index=6  Thumb=7
POS_A_EXCL = frozenset({3, 5})   # Position A exclusive keys
POS_B_EXCL = frozenset({2, 6})   # Position B exclusive keys

# Tendon-coupled pairs (shared extensor tendon → hard to alternate rapidly)
TENDON_PAIRS = {frozenset({1, 2}), frozenset({6, 7})}     # pinky–ring
MEDIUM_PAIRS = {frozenset({2, 3}), frozenset({5, 6})}     # ring–middle


# ── Chord ergonomic features ──────────────────────────────────────────────────

def chord_features(keys):
    """
    Static ergonomic cost features for a single-hand chord.
    keys: tuple of key positions (1–7), already sorted.
    """
    if not keys:
        return 0, 0, 0, 0, 0
    ks = set(keys)
    conflict  = int(bool(ks & POS_A_EXCL) and bool(ks & POS_B_EXCL))
    keys_s    = sorted(keys)
    adj_pairs = {frozenset({a, b}) for a, b in zip(keys_s, keys_s[1:]) if b == a + 1}
    n_adj     = len(adj_pairs)
    scr_side  = len(ks & {1, 2, 3})
    tendon    = len(adj_pairs & TENDON_PAIRS)
    medium_c  = len(adj_pairs & MEDIUM_PAIRS)
    return conflict, n_adj, scr_side, tendon, medium_c


def position_of(keys):
    ks = set(keys)
    if ks & POS_A_EXCL and not ks & POS_B_EXCL: return 'A'
    if ks & POS_B_EXCL and not ks & POS_A_EXCL: return 'B'
    return 'neutral'


def note_difficulty(keys, prev_keys, interval_rows, bpm, w):
    """
    Full per-chord difficulty on the θ scale.

    Structure:
        d = bias + (static_chord_cost + transition_cost) × time_pressure

    Wide chords (n≥5) are "mash zones" — the player lays the whole hand down and
    any hit counts for gauge purposes. Only timing matters, not finger placement.
    """
    if not keys:
        return w['bias']

    # Time pressure: how fast must the player execute this chord.
    # Cap interval at 12 rows (16th-note) — sub-16th gaps come from CN head/tap
    # overlaps and should not be treated as independent ultra-fast notes.
    interval_secs = max(interval_rows, 12) / (bpm / 60.0 * ROWS_PER_BEAT)
    notes_per_sec = 1.0 / interval_secs
    ref_nps       = (w['bpm_ref'] / 60.0) * 4   # 16th-note rate at reference BPM
    time_pressure = (notes_per_sec / ref_nps) ** w['bpm_exp']

    n = len(keys)
    if n >= 5:
        # Mash zone: just timing, not individual finger placement.
        return w['bias'] + w['w_wide'] * time_pressure

    conflict, n_adj, scr_side, _, _ = chord_features(keys)

    # Static: ergonomic cost of the chord shape itself.
    # Tendon/medium coupling are TRANSITION costs (affect rapid alternation,
    # not simultaneous pressing), so they are excluded here.
    static = (w['w_n']        * n
            + w['w_conflict'] * conflict
            + w['w_adj']      * n_adj
            + w['w_scr']      * scr_side)

    transition = 0.0
    if prev_keys:
        pos_switch  = int(position_of(prev_keys) != position_of(keys)
                         and 'neutral' not in [position_of(prev_keys), position_of(keys)])
        center_jump = abs(sum(keys)/len(keys) - sum(prev_keys)/len(prev_keys))

        # Tendon/medium trill: alternating use of coupled-key pairs across events.
        # Pressing both keys of a pair simultaneously is fine; rapid alternation is hard.
        prev_ks, cur_ks = set(prev_keys), set(keys)
        tendon_trill = sum(
            1 for pair in TENDON_PAIRS
            if (pair & prev_ks) and (pair & cur_ks) and (pair & prev_ks) != (pair & cur_ks)
        )
        medium_trill = sum(
            1 for pair in MEDIUM_PAIRS
            if (pair & prev_ks) and (pair & cur_ks) and (pair & prev_ks) != (pair & cur_ks)
        )

        transition = (w['w_switch'] * pos_switch
                    + w['w_reach']  * center_jump
                    + w['w_tendon'] * tendon_trill
                    + w['w_medium'] * medium_trill)

    return w['bias'] + (static + transition) * time_pressure


# ── Event extraction ──────────────────────────────────────────────────────────

def extract_events(chart):
    """
    Parse chart array into a list of active-note events.
    Returns list of (row, p1_keys, p2_keys, bpm, has_p1_scr, has_p2_scr).
    Only rows with at least one action note are included.
    BPM is filled forward from column 16.
    """
    n_rows = chart.shape[0]
    bpm_raw = chart[:, 16].astype(np.float64)

    # Fill-forward BPM
    cur_bpm = 150.0
    bpm_arr = np.empty(n_rows)
    for r in range(n_rows):
        if bpm_raw[r] > 0:
            cur_bpm = bpm_raw[r] / 100.0
        bpm_arr[r] = cur_bpm

    events = []
    for r in range(n_rows):
        p1 = tuple(l     for l in P1_LANES if int(chart[r, l]) in ACTION_VALUES)
        p2 = tuple(l - 7 for l in P2_LANES if int(chart[r, l]) in ACTION_VALUES)
        s1 = int(chart[r, 0])  in ACTION_VALUES
        s2 = int(chart[r, 15]) in ACTION_VALUES
        if p1 or p2 or s1 or s2:
            events.append((r, p1, p2, bpm_arr[r], int(s1), int(s2)))

    return events


# ── Gauge simulation ──────────────────────────────────────────────────────────

def simulate_gauges(events, w, phys, total_notes, chart_nps=10.0):
    """
    Simulate EC / HC / EXH gauges for all θ simultaneously.

    Uses expected-value update per note (vectorised over θ), equivalent to
    averaging over many independent runs of the same chart.

    chart_nps: global notes-per-second (both hands combined). Denser charts
    are harder due to sustained load; shifts all per-note difficulties up by
    min(w_density × max(0, chart_nps − ref_nps), w_density_cap).

    Returns:
        ec_pass, hc_pass, exh_pass — each shape (N_THETA,), values in [0, 1]
    """
    # EC recovery constant per note (depends on total note count)
    if total_notes < 350:
        a_ec = (80000.0 / (total_notes * 6))   / 50.0 / 100.0
    else:
        a_ec = (80000.0 / (total_notes * 2 + 1400)) / 50.0 / 100.0

    ref_nps = w['bpm_ref'] / 60.0 * 4
    density_offset = min(w['w_density'] * max(0.0, chart_nps - ref_nps),
                         w['w_density_cap'])

    N = len(THETA)
    ec_gauge  = np.full(N, 0.22)
    hc_gauge  = np.full(N, 1.00)
    exh_gauge = np.full(N, 1.00)
    hc_min    = np.ones(N)
    exh_min   = np.ones(N)

    alpha = w['alpha']

    prev_p1, prev_p2 = (), ()
    prev_row_p1 = prev_row_p2 = 0

    for row, p1, p2, bpm, s1, s2 in events:
        int_p1 = row - prev_row_p1 if p1 else ROWS_PER_BEAT
        int_p2 = row - prev_row_p2 if p2 else ROWS_PER_BEAT

        d_p1 = note_difficulty(p1, prev_p1, int_p1, bpm, w) if p1 else w['bias']
        d_p2 = note_difficulty(p2, prev_p2, int_p2, bpm, w) if p2 else w['bias']

        if p1: prev_p1, prev_row_p1 = p1, row
        if p2: prev_p2, prev_row_p2 = p2, row

        n_p1 = len(p1) + s1
        n_p2 = len(p2) + s2
        n_row = n_p1 + n_p2
        if n_row == 0:
            continue

        # Note-count-weighted difficulty for the row, plus chart-level density shift
        d_row = (d_p1 * n_p1 + d_p2 * n_p2) / n_row + density_offset

        # Hit probability per θ
        r = 1.0 / (1.0 + np.exp(-alpha * (THETA - d_row)))   # (N,)
        b = 1.0 - r

        # ── EC ────────────────────────────────────────────────────────────────
        ec_delta = (r * a_ec - b * phys['ec_drain']) * n_row
        ec_gauge = np.clip(ec_gauge + ec_delta, 0.0, 1.0)

        # ── HC (with 30% correction: BAD drain halved when gauge < 30%) ───────
        correction = np.where(hc_gauge < 0.30, 0.5, 1.0)
        hc_delta   = (r * phys['hc_rec'] - b * phys['hc_drain'] * correction) * n_row
        hc_gauge   = np.clip(hc_gauge + hc_delta, 0.0, 1.0)
        hc_min     = np.minimum(hc_min, hc_gauge)

        # ── EXH (no 30% correction) ───────────────────────────────────────────
        exh_delta  = (r * phys['hc_rec'] - b * phys['exh_drain']) * n_row
        exh_gauge  = np.clip(exh_gauge + exh_delta, 0.0, 1.0)
        exh_min    = np.minimum(exh_min, exh_gauge)

    ec_pass  = 1.0 / (1.0 + np.exp(-20.0 * (ec_gauge - 0.80)))
    hc_pass  = 1.0 / (1.0 + np.exp(-20.0 * hc_min))
    exh_pass = 1.0 / (1.0 + np.exp(-20.0 * exh_min))

    return ec_pass, hc_pass, exh_pass


def extract_theta_star(pass_prob):
    """θ* via derivative-weighted mean of the pass-probability curve."""
    dpass     = np.diff(pass_prob).clip(min=0.0)
    theta_mid = (THETA[1:] + THETA[:-1]) / 2.0
    total     = dpass.sum()
    if total < 1e-8:
        return float(THETA[-1]) if pass_prob.mean() < 0.5 else float(THETA[0])
    return float((dpass * theta_mid).sum() / total)


# ── Prediction pipeline ───────────────────────────────────────────────────────

def predict_chart(npy_path, w, phys):
    chart       = np.load(npy_path)
    events      = extract_events(chart)
    total_notes = sum(len(p1) + len(p2) + s1 + s2 for _, p1, p2, _, s1, s2 in events)
    if total_notes == 0:
        return (THETA[len(THETA)//2],) * 3
    # Compute global notes-per-second (both hands combined)
    if len(events) >= 2:
        first_row, last_row = events[0][0], events[-1][0]
        avg_bpm = float(np.mean([e[3] for e in events]))
        chart_secs = max((last_row - first_row) / (avg_bpm / 60.0 * ROWS_PER_BEAT), 1.0)
        chart_nps = total_notes / chart_secs
    else:
        chart_nps = w['bpm_ref'] / 60.0 * 4
    ec_pass, hc_pass, exh_pass = simulate_gauges(events, w, phys, total_notes, chart_nps)
    return (extract_theta_star(ec_pass),
            extract_theta_star(hc_pass),
            extract_theta_star(exh_pass))


def evaluate(manifest, w, phys, verbose=False):
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    preds, targets, titles = [], [], []
    for _, row in lv12.iterrows():
        npy = DP12_DIR / row['file_path']
        if not npy.exists():
            continue
        try:
            pred = predict_chart(str(npy), w, phys)
        except Exception as e:
            if verbose: print(f"  error {row['title']}: {e}")
            continue
        preds.append(pred)
        targets.append((float(row['rating_ec']),
                        float(row['rating_hc']),
                        float(row['rating_exh'])))
        titles.append(row['title'])

    return np.array(preds), np.array(targets), titles


def print_results(preds, targets, titles=None):
    print(f"\n── results ({len(preds)} charts) ─────────────────────────────────")
    for j, name in enumerate(['ec', 'hc', 'exh']):
        mae = np.mean(np.abs(preds[:, j] - targets[:, j]))
        rho = spearmanr(preds[:, j], targets[:, j]).statistic
        print(f"  {name}: MAE={mae:.3f}  ρ={rho:.4f}  "
              f"pred=[{preds[:,j].min():.1f}, {preds[:,j].max():.1f}]  "
              f"target=[{targets[:,j].min():.1f}, {targets[:,j].max():.1f}]")

    if titles is not None:
        ec_err = np.abs(preds[:, 0] - targets[:, 0])
        worst  = np.argsort(ec_err)[-10:][::-1]
        print(f"\n── worst EC predictions ─────────────────────────────────────")
        for i in worst:
            print(f"  {titles[i]:<40}  ec={targets[i,0]:.1f}  "
                  f"pred={preds[i,0]:.1f}  err={ec_err[i]:.1f}")


# ── Default parameters ────────────────────────────────────────────────────────

WEIGHTS = dict(
    bias      = 0.94,  # base difficulty floor
    w_n       = 1.07,  # cost per simultaneous note (n ≤ 4)
    w_conflict= 2.00,  # position conflict (needs both Pos A and Pos B keys)
    w_adj     = 1.00,  # adjacent key pair
    w_scr     = 0.01,  # scratch-side key (near zero — not a strong predictor)
    w_switch  = 0.92,  # position switch from previous chord
    w_reach   = 0.00,  # per-unit hand centre jump (near zero)
    w_tendon  = 0.00,  # tendon trill penalty (near zero in optimised weights)
    w_medium  = 0.61,  # ring–middle trill penalty
    w_wide    = 0.02,  # wide chord (n≥5) difficulty — mash zone, nearly free
    w_density = 0.055, # chart density penalty per nps above reference
    w_density_cap = 6.2, # hard cap on density offset
    alpha     = 1.2,   # IRT sharpness
    bpm_ref   = 200.0, # reference BPM (higher → less time pressure at typical BPM)
    bpm_exp   = 0.7,   # BPM scaling exponent
)

PHYSICS = dict(
    ec_drain  = 0.048,          # EC: BAD drains 4.8% of gauge (fraction: 0.048)
    hc_rec    = 0.0016,         # HC/EXH: PGREAT/GREAT recovers 0.16%
    hc_drain  = 0.05,           # HC: BAD drains 5%
    exh_drain = 0.10,           # EXH: BAD drains 10%
)


if __name__ == '__main__':
    import time
    manifest = pd.read_csv(MANIFEST)

    print("Simulator — running on all lv12 charts...")
    t0 = time.time()
    preds, targets, titles = evaluate(manifest, WEIGHTS, PHYSICS)
    elapsed = time.time() - t0
    print(f"  {len(preds)} charts in {elapsed:.1f}s ({elapsed/len(preds)*1000:.0f}ms/chart)")
    print_results(preds, targets, titles)
