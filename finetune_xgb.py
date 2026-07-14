"""
XGBoost probes on half-chart encoder embeddings.

Approach A: XGBoost on PCA(128)+note features (same as Ridge baseline, non-linear test)
Approach B: XGBoost on per-segment PCA(16) features — 10 segs × 16-dim = 160 + 8 note = 168 features
            Tests whether XGBoost can learn temporal segment interactions.

Baseline comparison: k=5 weighted mean + Ridge  (EC ρ=0.772, HC ρ=0.774, EXH ρ=0.737)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from xgboost import XGBRegressor

from dataset import (
    encode_chart_pretrain_v6, window_chart_split,
    WINDOW_BARS, STRIDE_BARS,
    PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
    NUM_PATCHES_V4, PATCH_ROWS_V4,
)
from mae import MAEModel

# ── Config ────────────────────────────────────────────────────────────────────
MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
CKPT      = Path('checkpoints/pretrain_half/encoder_best.pt')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGETS   = ['rating_ec', 'rating_hc', 'rating_exh']
EMBED_DIM = 256
N_SEGS    = 10
SEG_PCA   = 16    # per-segment PCA dim for approach B
K_WEIGHT  = 5     # best k from weighted-mean experiments

RIDGE_RHO = {'rating_ec': 0.7722, 'rating_hc': 0.7747, 'rating_exh': 0.7376}  # k=5
RIDGE_MAE = {'rating_ec': 1.3888, 'rating_hc': 1.0200, 'rating_exh': 0.7780}
XGB_BASE_RHO = {'rating_ec': 0.7303, 'rating_hc': 0.7316, 'rating_exh': 0.6920}

XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method='hist',
    random_state=42,
    verbosity=0,
)


# ── Encoder ───────────────────────────────────────────────────────────────────

def load_encoder():
    model = MAEModel(
        in_channels=PRETRAIN_IN_CHANNELS_HALF,
        num_patches=NUM_PATCHES_V4, patch_rows=PATCH_ROWS_V4,
        encoder_dim=EMBED_DIM, decoder_dim=128, num_heads=8,
        use_lane_type_embed=True, use_bpm_cond=True,
        lane_types=LANE_TYPES_HALF,
    )
    model.encoder.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    return model.encoder.eval().to(DEVICE)


@torch.no_grad()
def embed_windows(encoder, wins):
    x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
    bpm = torch.from_numpy(wins[:, :, 8].mean(axis=1)).float().to(DEVICE)
    return encoder.embed_segment(x, bpm=bpm).cpu().numpy()  # (T, 256)


def weighted_mean(cls, k):
    T = len(cls)
    if T == 1 or k == 0:
        return cls.mean(0)
    t = np.arange(T, dtype=np.float32)
    w = 1.0 + k * (t / (T - 1))
    w /= w.sum()
    return (cls * w[:, None]).sum(0)


# ── Pre-compute ───────────────────────────────────────────────────────────────

def precompute(encoder, manifest):
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        manifest['rating_hc'].notna() &
        manifest['rating_exh'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    flat_embs   = []   # (N_charts, 512)  — k=5 weighted fused mean
    seg_embs    = []   # (N_charts, N_SEGS, 512)  — per-segment fused means
    note_feats  = []
    labels      = []
    skipped = 0

    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='embedding'):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            skipped += 1; continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        p1_wins, p2_wins, win_nc = window_chart_split(enc, nc, WINDOW_BARS, STRIDE_BARS)
        if p1_wins.shape[0] == 0:
            skipped += 1; continue

        p1_cls = embed_windows(encoder, p1_wins)
        p2_cls = embed_windows(encoder, p2_wins)

        # approach A: k=5 weighted fused mean → (512,)
        p1_w = weighted_mean(p1_cls, K_WEIGHT)
        p2_w = weighted_mean(p2_cls, K_WEIGHT)
        flat_embs.append(np.concatenate([(p1_w + p2_w) / 2.0, p1_w - p2_w]))

        # approach B: per-segment fused means → (N_SEGS, 512)
        p1_parts = np.array_split(p1_cls, N_SEGS)
        p2_parts = np.array_split(p2_cls, N_SEGS)
        segs = []
        for p1s, p2s in zip(p1_parts, p2_parts):
            m1, m2 = p1s.mean(0), p2s.mean(0)
            segs.append(np.concatenate([(m1 + m2) / 2.0, m1 - m2]))
        seg_embs.append(np.stack(segs))   # (N_SEGS, 512)

        nc_arr = win_nc.astype(np.float32)
        last_q = nc_arr[max(0, int(len(nc_arr) * 0.75)):]
        note_feats.append([
            nc_arr.mean(), nc_arr.std() + 1e-6, nc_arr.max(),
            np.percentile(nc_arr, 75), np.percentile(nc_arr, 25),
            last_q.mean() / (nc_arr.mean() + 1e-6),
            (nc_arr.std() + 1e-6) / (nc_arr.mean() + 1e-6),
            nc_arr.sum(),
        ])
        labels.append([float(row['rating_ec']), float(row['rating_hc']), float(row['rating_exh'])])

    print(f'embedded {len(flat_embs)} charts  (skipped {skipped})', flush=True)
    return (np.stack(flat_embs),
            np.stack(seg_embs),
            np.array(note_feats),
            np.array(labels))


def run_cv(X, y, label):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = {t: [] for t in TARGETS}
    rhos = {t: [] for t in TARGETS}
    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        preds = np.zeros_like(y_va)
        for j, t in enumerate(TARGETS):
            m = XGBRegressor(**XGB_PARAMS)
            m.fit(X_tr, y_tr[:, j])
            preds[:, j] = m.predict(X_va)
        parts = [f'  {label} fold {fold}:']
        for j, t in enumerate(TARGETS):
            mae = float(np.abs(preds[:, j] - y_va[:, j]).mean())
            rho = float(spearmanr(y_va[:, j], preds[:, j]).statistic)
            maes[t].append(mae); rhos[t].append(rho)
            parts.append(f'  {t.split("_")[1].upper()} MAE={mae:.4f} ρ={rho:.3f}')
        print(''.join(parts), flush=True)
    return maes, rhos


# ── Main ──────────────────────────────────────────────────────────────────────

print(f'device: {DEVICE}', flush=True)
encoder  = load_encoder()
manifest = pd.read_csv(MANIFEST)
flat_embs, seg_embs, X_note, y = precompute(encoder, manifest)

# ── Approach A: XGBoost on PCA(128) + note features ──────────────────────────
print('\n── Approach A: XGBoost on PCA(128) + note-profile ──────────────────────', flush=True)
scaler_a   = StandardScaler()
X_flat_s   = scaler_a.fit_transform(flat_embs)
X_pca_a    = PCA(n_components=128, random_state=42).fit_transform(X_flat_s)
note_s     = StandardScaler().fit_transform(X_note)
X_a        = np.concatenate([X_pca_a, note_s], axis=1)   # (N, 136)
print(f'features: {X_a.shape}', flush=True)
maes_a, rhos_a = run_cv(X_a, y, 'A')

# ── Approach B: XGBoost on per-segment PCA(16) + note features ───────────────
print('\n── Approach B: XGBoost on per-segment PCA(16) + note-profile ───────────', flush=True)
# fit one PCA per segment position across all charts
N, S, D = seg_embs.shape   # (684, 10, 512)
seg_reduced = np.zeros((N, S * SEG_PCA), dtype=np.float32)
for s in range(S):
    seg_s = StandardScaler().fit_transform(seg_embs[:, s, :])
    seg_reduced[:, s*SEG_PCA:(s+1)*SEG_PCA] = \
        PCA(n_components=SEG_PCA, random_state=42).fit_transform(seg_s)
X_b = np.concatenate([seg_reduced, note_s], axis=1)   # (N, 168)
print(f'features: {X_b.shape}  ({S} segs × PCA({SEG_PCA}) + 8 note)', flush=True)
maes_b, rhos_b = run_cv(X_b, y, 'B')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n── Results ─────────────────────────────────────────────────────────────', flush=True)
print(f'{"target":<12}  {"XGB-base":>9}  {"Ridge k=5":>10}  {"A: XGB-PCA":>11}  {"B: XGB-seg":>11}')
print('─' * 65)
for t in TARGETS:
    tname = t.split('_')[1].upper()
    ra = np.mean(rhos_a[t]); rb = np.mean(rhos_b[t])
    ma = np.mean(maes_a[t]); mb = np.mean(maes_b[t])
    print(f'{t:<12}  ρ  {XGB_BASE_RHO[t]:9.4f}  {RIDGE_RHO[t]:10.4f}  {ra:11.4f}  {rb:11.4f}')
    print(f'{"":12}  MAE{"":6}  {"":9}  {RIDGE_MAE[t]:10.4f}  {ma:11.4f}  {mb:11.4f}')
