"""
GBT baseline: handcrafted chart features → rating_ec / rating_hc / rating_exh.
Evaluation metric: MAE on each target (5-fold CV) + Spearman ρ.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
import xgboost as xgb

# ── Config ────────────────────────────────────────────────────────────────────
MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
TARGETS   = ['rating_ec', 'rating_hc', 'rating_exh']
ROWS_PER_BAR = 192
SCRATCH_LANES = {0, 15}
KEY_LANES     = set(range(1, 15))


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(npy_path: Path, json_path: Path) -> dict | None:
    try:
        arr  = np.load(npy_path).astype(np.float32)   # (total_rows, 17)
        meta = json.loads(json_path.read_text())
    except Exception:
        return None

    total_rows = arr.shape[0]
    lanes = arr[:, :16]   # (total_rows, 16)
    bpm_col = arr[:, 16]  # BPM×100, fill-forward

    # ── BPM ──────────────────────────────────────────────────────────────────
    bpm_raw = bpm_col.copy()
    last = 0.0
    for i in range(len(bpm_raw)):
        if bpm_raw[i] > 0:
            last = bpm_raw[i]
        bpm_raw[i] = last
    bpm = bpm_raw / 100.0  # actual BPM per row

    bpm_nonzero = bpm[bpm > 0]
    bpm_mean = float(bpm_nonzero.mean()) if len(bpm_nonzero) else 120.0
    bpm_max  = float(bpm_nonzero.max())  if len(bpm_nonzero) else 120.0
    bpm_changes = int((bpm_col > 0).sum()) - 1  # first row doesn't count as change

    # ── Note masks ───────────────────────────────────────────────────────────
    HEAD_VALS = {1, 2, 4, 6, 8}   # tap + CN/HCN/BSS/MSS heads
    BODY_VALS = {3, 5, 7, 9}      # hold bodies

    head_mask = np.isin(lanes.astype(np.int32), list(HEAD_VALS))  # (rows, 16)
    body_mask = np.isin(lanes.astype(np.int32), list(BODY_VALS))
    note_mask = head_mask | body_mask  # any active note (head or body)

    # ── Total note count (heads only — motor events) ─────────────────────────
    total_notes = int(head_mask.sum())

    # ── Chart duration ───────────────────────────────────────────────────────
    total_bars = meta.get('total_bars', total_rows // ROWS_PER_BAR)
    dur_beats  = total_bars * 4.0
    dur_sec    = dur_beats / bpm_mean * 60.0

    # ── Per-bar density (heads only) ─────────────────────────────────────────
    n_bars = total_rows // ROWS_PER_BAR
    bar_counts = np.array([
        head_mask[i*ROWS_PER_BAR:(i+1)*ROWS_PER_BAR].sum()
        for i in range(n_bars)
    ], dtype=np.float32)

    density_mean = float(bar_counts.mean())
    density_max  = float(bar_counts.max())
    density_std  = float(bar_counts.std())
    notes_per_sec = total_notes / max(dur_sec, 1.0)

    # ── Density peaks: top-10% bars ──────────────────────────────────────────
    peak_threshold = np.percentile(bar_counts, 90)
    peak_density_mean = float(bar_counts[bar_counts >= peak_threshold].mean())

    # ── Hard ending: last 25% of bars ────────────────────────────────────────
    ending_start = int(n_bars * 0.75)
    ending_counts = bar_counts[ending_start:]
    ending_density_mean = float(ending_counts.mean()) if len(ending_counts) else 0.0
    ending_density_max  = float(ending_counts.max())  if len(ending_counts) else 0.0
    ending_ratio = ending_density_mean / max(density_mean, 1.0)

    # ── Chord features (per active row) ──────────────────────────────────────
    row_note_counts = head_mask.sum(axis=1)          # notes per row
    active_rows = row_note_counts > 0
    n_active = int(active_rows.sum())

    chord_mean = float(row_note_counts[active_rows].mean()) if n_active else 0.0
    chord_max  = int(row_note_counts.max())
    chord_3plus = float((row_note_counts >= 3).sum()) / max(total_rows, 1)
    chord_4plus = float((row_note_counts >= 4).sum()) / max(total_rows, 1)
    active_row_rate = n_active / max(total_rows, 1)

    # ── Scratch features ─────────────────────────────────────────────────────
    sc0  = head_mask[:, 0]   # P1 scratch
    sc15 = head_mask[:, 15]  # P2 scratch
    keys = head_mask[:, 1:15]

    scratch_rate      = float((sc0 | sc15).mean())
    double_scratch    = float((sc0 & sc15).mean())
    scratch_key_rate  = float(((sc0 | sc15) & (keys.sum(axis=1) > 0)).mean())
    p1_scratch_rate   = float(sc0.mean())
    p2_scratch_rate   = float(sc15.mean())

    # ── Effective density (BPM-normalised) ───────────────────────────────────
    effective_density = notes_per_sec * (bpm_mean / 150.0)

    # ── Chart-level flags ─────────────────────────────────────────────────────
    has_hcn = int(meta.get('hcn', False))
    has_bss = int(meta.get('has_bss', False))
    has_mss = int(meta.get('has_mss', False))

    return {
        # density
        'total_notes':        total_notes,
        'notes_per_sec':      notes_per_sec,
        'density_mean':       density_mean,
        'density_max':        density_max,
        'density_std':        density_std,
        'effective_density':  effective_density,
        'peak_density_mean':  peak_density_mean,
        'active_row_rate':    active_row_rate,
        # chords
        'chord_mean':         chord_mean,
        'chord_max':          chord_max,
        'chord_3plus_rate':   chord_3plus,
        'chord_4plus_rate':   chord_4plus,
        # scratch
        'scratch_rate':       scratch_rate,
        'double_scratch_rate': double_scratch,
        'scratch_key_rate':   scratch_key_rate,
        'p1_scratch_rate':    p1_scratch_rate,
        'p2_scratch_rate':    p2_scratch_rate,
        # ending (EC-relevant)
        'ending_density_mean': ending_density_mean,
        'ending_density_max':  ending_density_max,
        'ending_ratio':        ending_ratio,
        # BPM
        'bpm_mean':           bpm_mean,
        'bpm_max':            bpm_max,
        'bpm_changes':        bpm_changes,
        # chart-level
        'total_bars':         total_bars,
        'has_hcn':            has_hcn,
        'has_bss':            has_bss,
        'has_mss':            has_mss,
    }


# ── Load data ─────────────────────────────────────────────────────────────────

manifest = pd.read_csv(MANIFEST)
lv12 = manifest[
    (manifest['level'] == 12) &
    manifest['rating_ec'].notna() &
    (manifest['status'] == 'ok')
].copy().reset_index(drop=True)

print(f"lv12 charts: {len(lv12)}")

rows, indices = [], []
for i, row in lv12.iterrows():
    npy  = DATA_ROOT / 'dp12_active' / str(row['file_path'])
    json_p = npy.with_suffix('.json')
    feats = extract_features(npy, json_p)
    if feats is None:
        continue
    rows.append(feats)
    indices.append(i)

X = pd.DataFrame(rows)
lv12 = lv12.loc[indices].reset_index(drop=True)
print(f"Features extracted: {len(X)} charts  |  {X.shape[1]} features")
print(f"Feature names: {list(X.columns)}")

# ── Train & evaluate ──────────────────────────────────────────────────────────

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for target in TARGETS:
    y = lv12[target].values.astype(np.float32)
    naive_mae = float(np.abs(y - y.mean()).mean())

    fold_maes, fold_rhos = [], []
    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
        pred = model.predict(X_val)
        fold_maes.append(float(np.abs(pred - y_val).mean()))
        fold_rhos.append(float(spearmanr(y_val, pred).statistic))

    mae_mean = np.mean(fold_maes)
    mae_std  = np.std(fold_maes)
    rho_mean = np.mean(fold_rhos)

    improvement = (naive_mae - mae_mean) / naive_mae * 100
    print(f"\n{target}:")
    print(f"  naive MAE:  {naive_mae:.4f}")
    print(f"  XGB MAE:    {mae_mean:.4f} ± {mae_std:.4f}  ({improvement:+.1f}% vs naive)")
    print(f"  Spearman ρ: {rho_mean:.4f}")

# ── Feature importance (full fit on rating_hc) ────────────────────────────────
print("\nFeature importance (XGB, full fit on rating_hc):")
y_hc = lv12['rating_hc'].values.astype(np.float32)
full_model = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                               subsample=0.8, colsample_bytree=0.8,
                               reg_lambda=1.0, random_state=42, verbosity=0)
full_model.fit(X, y_hc)
imp = pd.Series(full_model.feature_importances_, index=X.columns)
print(imp.sort_values(ascending=False).to_string())
