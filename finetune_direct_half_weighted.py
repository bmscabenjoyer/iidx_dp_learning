"""
Frozen half-chart encoder probe with position-weighted mean pooling.
w(t) = 1 + k * (t / T) — later windows get more weight.
Tests k in [0, 1, 2, 3] against flat mean (k=0) to check whether
weighting toward hard endings improves predictions.
Same 512-dim [mean,diff] fusion and PCA(128) pipeline throughout.
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
PCA_DIM   = 128
K_VALUES  = [0, 1, 2, 3]   # k=0 reproduces flat mean

HALF_RHO  = {'rating_ec': 0.7514, 'rating_hc': 0.7605, 'rating_exh': 0.7203}
HALF_MAE  = {'rating_ec': 1.4576, 'rating_hc': 1.0507, 'rating_exh': 0.8061}
XGB_RHO   = {'rating_ec': 0.7303, 'rating_hc': 0.7316, 'rating_exh': 0.6920}
XGB_MAE   = {'rating_ec': 1.4570, 'rating_hc': 1.0998, 'rating_exh': 0.8363}


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
    """Returns per-window CLS embeddings, shape (T, EMBED_DIM)."""
    x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
    bpm = torch.from_numpy(wins[:, :, 8].mean(axis=1)).float().to(DEVICE)
    return encoder.embed_segment(x, bpm=bpm).cpu().numpy()  # (T, 256)


def weighted_mean(cls, k):
    """Weighted mean: w(t) = 1 + k * (t / (T-1)), normalised to sum=1."""
    T = len(cls)
    if T == 1 or k == 0:
        return cls.mean(0)
    t = np.arange(T, dtype=np.float32)
    w = 1.0 + k * (t / (T - 1))
    w /= w.sum()
    return (cls * w[:, None]).sum(0)


def precompute(encoder, manifest):
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        manifest['rating_hc'].notna() &
        manifest['rating_exh'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    # store raw per-window embeddings so we can apply different k without re-encoding
    all_p1_cls, all_p2_cls = [], []
    note_feats, labels = [], []
    skipped = 0

    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='embedding charts'):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            skipped += 1
            continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        p1_wins, p2_wins, win_nc = window_chart_split(enc, nc, WINDOW_BARS, STRIDE_BARS)
        if p1_wins.shape[0] == 0:
            skipped += 1
            continue

        all_p1_cls.append(embed_windows(encoder, p1_wins))  # (T, 256)
        all_p2_cls.append(embed_windows(encoder, p2_wins))  # (T, 256)

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

    print(f'embedded {len(all_p1_cls)} charts  (skipped {skipped})', flush=True)
    return all_p1_cls, all_p2_cls, np.array(note_feats), np.array(labels)


def build_features(all_p1, all_p2, X_note, k, pca_dim=PCA_DIM):
    embs = []
    for p1_cls, p2_cls in zip(all_p1, all_p2):
        p1 = weighted_mean(p1_cls, k)
        p2 = weighted_mean(p2_cls, k)
        embs.append(np.concatenate([(p1 + p2) / 2.0, p1 - p2]))  # (512,)
    X_emb = np.stack(embs)
    X_emb_s = StandardScaler().fit_transform(X_emb)
    X_pca   = PCA(n_components=pca_dim, random_state=42).fit_transform(X_emb_s)
    X_note_s = StandardScaler().fit_transform(X_note)
    return np.concatenate([X_pca, X_note_s], axis=1)


def run_cv(X, y, alphas):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = {t: [] for t in TARGETS}
    rhos = {t: [] for t in TARGETS}
    for tr_idx, va_idx in kf.split(X):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        preds = np.zeros_like(y_va)
        for j in range(len(TARGETS)):
            r = RidgeCV(alphas=alphas, cv=5)
            r.fit(X_tr, y_tr[:, j])
            preds[:, j] = r.predict(X_va)
        for j, t in enumerate(TARGETS):
            maes[t].append(float(np.abs(preds[:, j] - y_va[:, j]).mean()))
            rhos[t].append(float(spearmanr(y_va[:, j], preds[:, j]).statistic))
    return maes, rhos


# ── Main ──────────────────────────────────────────────────────────────────────

print(f'device: {DEVICE}', flush=True)
encoder  = load_encoder()
manifest = pd.read_csv(MANIFEST)
all_p1, all_p2, X_note, y = precompute(encoder, manifest)

alphas = np.logspace(-1, 5, 40)
results = {}

for k in K_VALUES:
    print(f'\n── k={k}  (w(t) = 1 + {k}×t/T) ──', flush=True)
    X = build_features(all_p1, all_p2, X_note, k)
    maes, rhos = run_cv(X, y, alphas)
    results[k] = (maes, rhos)
    for t in TARGETS:
        print(f'  {t.split("_")[1].upper()}  MAE={np.mean(maes[t]):.4f}  ρ={np.mean(rhos[t]):.4f}',
              flush=True)

print('\n── Summary ──────────────────────────────────────────────────────────────', flush=True)
print(f'{"":12}  ' + '  '.join(f'k={k:1d}  ρ' for k in K_VALUES))
for t in TARGETS:
    tname = t.split('_')[1].upper()
    row = f'{tname:<12}'
    for k in K_VALUES:
        _, rhos = results[k]
        row += f'   {np.mean(rhos[t]):.4f}'
    row += f'   (flat-mean: {HALF_RHO[t]:.4f})'
    print(row, flush=True)

print('\n── MAE ──────────────────────────────────────────────────────────────────', flush=True)
for t in TARGETS:
    tname = t.split('_')[1].upper()
    row = f'{tname:<12}'
    for k in K_VALUES:
        maes, _ = results[k]
        row += f'   {np.mean(maes[t]):.4f}'
    row += f'   (flat-mean: {HALF_MAE[t]:.4f})'
    print(row, flush=True)
