"""
Frozen v7 half-chart encoder probe: time-based windows (8s, 150 BPM ref)
P1+P2 embeddings fused as [mean, diff] → PCA(128) + 8 note-profile → RidgeCV
5-fold CV vs v6 half encoder baseline.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dataset import (
    encode_chart_pretrain_v6, window_chart_time_half,
    PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
    NUM_PATCHES_V7, PATCH_ROWS_V7,
)
from mae import MAEModel

# ── Config ────────────────────────────────────────────────────────────────────
MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
CKPT      = Path('checkpoints/pretrain_half_v7/encoder_best.pt')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGETS   = ['rating_ec', 'rating_hc', 'rating_exh']
EMBED_DIM = 256
PCA_DIM   = 128

# baselines from previous runs
XGB_MAE  = {'rating_ec': 1.4570, 'rating_hc': 1.0998, 'rating_exh': 0.8363}
XGB_RHO  = {'rating_ec': 0.7303, 'rating_hc': 0.7316, 'rating_exh': 0.6920}
HALF_RHO = {'rating_ec': 0.7511, 'rating_hc': 0.7258, 'rating_exh': 0.6872}
HALF_MAE = {'rating_ec': 1.4608, 'rating_hc': 1.1140, 'rating_exh': 0.8377}


def load_encoder():
    model = MAEModel(
        in_channels=PRETRAIN_IN_CHANNELS_HALF,
        num_patches=NUM_PATCHES_V7, patch_rows=PATCH_ROWS_V7,
        encoder_dim=EMBED_DIM, decoder_dim=128, num_heads=8,
        use_lane_type_embed=True, use_bpm_cond=True,
        lane_types=LANE_TYPES_HALF,
    )
    model.encoder.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    return model.encoder.eval().to(DEVICE)


@torch.no_grad()
def precompute(encoder, manifest):
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        manifest['rating_hc'].notna() &
        manifest['rating_exh'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    embeddings, note_feats, labels = [], [], []
    skipped = 0
    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='embedding charts'):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            skipped += 1
            continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        p1_wins, p2_wins, win_nc = window_chart_time_half(enc, nc)
        if p1_wins.shape[0] == 0:
            skipped += 1
            continue

        def embed_hand(wins):
            x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
            bpm = torch.from_numpy(wins[:, :, 8].mean(axis=1)).float().to(DEVICE)
            return encoder.embed_segment(x, bpm=bpm).mean(0).cpu().numpy()  # (256,)

        p1_emb = embed_hand(p1_wins)
        p2_emb = embed_hand(p2_wins)
        fused  = np.concatenate([(p1_emb + p2_emb) / 2.0, p1_emb - p2_emb])  # (512,)
        embeddings.append(fused)

        nc_arr = win_nc.astype(np.float32)
        last_q = nc_arr[max(0, int(len(nc_arr) * 0.75)):]
        note_feats.append([
            nc_arr.mean(),
            nc_arr.std() + 1e-6,
            nc_arr.max(),
            np.percentile(nc_arr, 75),
            np.percentile(nc_arr, 25),
            last_q.mean() / (nc_arr.mean() + 1e-6),
            (nc_arr.std() + 1e-6) / (nc_arr.mean() + 1e-6),
            nc_arr.sum(),
        ])
        labels.append([float(row['rating_ec']), float(row['rating_hc']), float(row['rating_exh'])])

    print(f'embedded {len(embeddings)} charts  (skipped {skipped})', flush=True)
    return np.stack(embeddings), np.array(note_feats), np.array(labels)


# ── Main ──────────────────────────────────────────────────────────────────────

print(f'device: {DEVICE}', flush=True)
encoder  = load_encoder()
manifest = pd.read_csv(MANIFEST)
X_emb, X_note, y = precompute(encoder, manifest)
print(f'X_emb: {X_emb.shape},  X_note: {X_note.shape},  y: {y.shape}', flush=True)

emb_scaler  = StandardScaler()
note_scaler = StandardScaler()
X_emb_s     = emb_scaler.fit_transform(X_emb)
pca         = PCA(n_components=PCA_DIM, random_state=42)
X_pca       = pca.fit_transform(X_emb_s)
X_note_s    = note_scaler.fit_transform(X_note)
X_combined  = np.concatenate([X_pca, X_note_s], axis=1)   # (N, 136)
print(f'combined features: {X_combined.shape}  (PCA {PCA_DIM} + note-profile 8)', flush=True)

kf     = KFold(n_splits=5, shuffle=True, random_state=42)
alphas = np.logspace(-1, 5, 40)

ridge_maes = {t: [] for t in TARGETS}
ridge_rhos = {t: [] for t in TARGETS}

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_combined), 1):
    print(f'\n── fold {fold} ──', flush=True)
    X_tr, X_va = X_combined[tr_idx], X_combined[va_idx]
    y_tr, y_va = y[tr_idx],          y[va_idx]

    ridge_preds = np.zeros_like(y_va)
    for j, t in enumerate(TARGETS):
        r = RidgeCV(alphas=alphas, cv=5)
        r.fit(X_tr, y_tr[:, j])
        ridge_preds[:, j] = r.predict(X_va)
        if fold == 1:
            print(f'  {t}: best alpha={r.alpha_:.1f}', flush=True)

    parts = [f'Ridge  fold {fold}:']
    for j, t in enumerate(TARGETS):
        p   = ridge_preds[:, j]
        yv  = y_va[:, j]
        mae = float(np.abs(p - yv).mean())
        rho = float(spearmanr(yv, p).statistic)
        ridge_maes[t].append(mae)
        ridge_rhos[t].append(rho)
        parts.append(f'  {t.split("_")[1].upper()} MAE={mae:.4f} ρ={rho:.3f}')
    print(''.join(parts), flush=True)

print('\n── Results ─────────────────────────────────────────────────────────────', flush=True)
print(f'{"target":<12}  {"naive":>7}  {"XGB ρ":>7}  {"half ρ":>7}  {"v7 ρ":>7}  '
      f'{"XGB MAE":>8}  {"half MAE":>9}  {"v7 MAE":>8}')
print('─' * 85)
for j, t in enumerate(TARGETS):
    naive = float(np.abs(y[:, j] - y[:, j].mean()).mean())
    mae   = np.mean(ridge_maes[t])
    mae_s = np.std(ridge_maes[t])
    rho   = np.mean(ridge_rhos[t])
    print(f'{t:<12}  {naive:7.4f}  {XGB_RHO[t]:7.4f}  {HALF_RHO[t]:7.4f}  {rho:7.4f}  '
          f'{XGB_MAE[t]:8.4f}  {HALF_MAE[t]:9.4f}  {mae:6.4f}±{mae_s:.4f}', flush=True)
