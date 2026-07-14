"""
Encoder evaluation:
  1. Linear probe — frozen CLS embeddings -> rating_stat (MAE, Spearman)
  2. Reconstruction precision/recall on note cells
  3. Nearest-neighbour sanity check (do similar-difficulty charts cluster?)
"""

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

from dataset import (
    encode_chart_pretrain_v6, window_chart,
    WINDOW_BARS, STRIDE_BARS, ROWS_PER_BAR,
    PRETRAIN_IN_CHANNELS_V6, NUM_PATCHES_V4, PATCH_ROWS_V4, BPM_SCALE
)
from mae import MAEModel, MAEEncoder

# ── Config ────────────────────────────────────────────────────────────────────
MANIFEST   = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT  = Path('/home/jysuh/projects/iidx_data')
CKPT       = Path('checkpoints/pretrain_v6/encoder_best.pt')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LEVEL_DIRS = {10: 'dp10_active', 11: 'dp11_active', 12: 'dp12_active'}

IN_CH = PRETRAIN_IN_CHANNELS_V6  # 16


def load_encoder() -> MAEEncoder:
    model = MAEModel(in_channels=IN_CH, num_patches=NUM_PATCHES_V4, patch_rows=PATCH_ROWS_V4,
                     encoder_dim=256, decoder_dim=128, num_heads=8,
                     use_lane_type_embed=True, use_bpm_cond=True)
    model.encoder.load_state_dict(torch.load(CKPT, map_location=DEVICE))
    model.encoder.eval().to(DEVICE)
    return model.encoder


@torch.no_grad()
def chart_embedding(encoder: MAEEncoder, npy_path: Path) -> np.ndarray | None:
    """Mean-pool CLS tokens across all windows → (embed_dim,) chart embedding."""
    enc, nc, _ = encode_chart_pretrain_v6(str(npy_path))
    windows, _ = window_chart(enc, nc, WINDOW_BARS, STRIDE_BARS)
    if windows.shape[0] == 0:
        return None
    x   = torch.from_numpy(windows[:, :, :IN_CH]).float().to(DEVICE)   # (T, 768, 16)
    bpm = torch.from_numpy(windows[:, :, 16].mean(axis=1)).float().to(DEVICE)  # (T,) normalised
    cls = encoder.embed_segment(x, bpm=bpm)   # (T, 256)
    return cls.mean(0).cpu().numpy()           # (256,)


# ── 1. Load labeled lv12 charts ───────────────────────────────────────────────
PROBE_TARGET = 'rating_exh'

manifest = pd.read_csv(MANIFEST)
lv12 = manifest[
    (manifest['level'] == 12) &
    manifest[PROBE_TARGET].notna() &
    (manifest['status'] == 'ok')
].copy().reset_index(drop=True)

print(f"lv12 charts with {PROBE_TARGET}: {len(lv12)}")

encoder = load_encoder()
print(f"Encoder loaded from {CKPT}")

embeddings, labels = [], []
skipped = 0
for _, row in lv12.iterrows():
    npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
    if not npy.exists():
        skipped += 1
        continue
    emb = chart_embedding(encoder, npy)
    if emb is None:
        skipped += 1
        continue
    embeddings.append(emb)
    labels.append(float(row[PROBE_TARGET]))

print(f"Embedded {len(embeddings)} charts  (skipped {skipped})")

X = np.stack(embeddings)   # (N, 128)
y = np.array(labels)       # (N,)

print(f"\nlabel stats:  mean={y.mean():.3f}  std={y.std():.3f}  "
      f"min={y.min():.2f}  max={y.max():.2f}")
print(f"naive MAE (predict mean): {np.abs(y - y.mean()).mean():.4f}")

# ── 2. Linear probe (5-fold cross-val) ───────────────────────────────────────
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

ridge = Ridge(alpha=1.0)
neg_mae = cross_val_score(ridge, X_s, y, cv=5, scoring='neg_mean_absolute_error')
mae_scores = -neg_mae
print(f"\nLinear probe (Ridge, 5-fold CV):")
print(f"  MAE per fold: {mae_scores.round(4)}")
print(f"  mean MAE:     {mae_scores.mean():.4f} ± {mae_scores.std():.4f}")

# Spearman on full fit (train=all, for ranking quality)
ridge.fit(X_s, y)
pred = ridge.predict(X_s)
rho, p = spearmanr(y, pred)
print(f"  Spearman ρ (train-fit): {rho:.4f}  (p={p:.2e})")

# ── 3. Nearest-neighbour check ────────────────────────────────────────────────
print("\nNearest-neighbour sanity check (5 random queries):")
rng = np.random.default_rng(42)
query_ids = rng.choice(len(embeddings), size=5, replace=False)

norms = np.linalg.norm(X, axis=1, keepdims=True)
X_norm = X / (norms + 1e-8)

for qi in query_ids:
    sims = X_norm @ X_norm[qi]
    sims[qi] = -1  # exclude self
    nn_id = int(sims.argmax())
    q_row  = lv12.iloc[[i for i, _ in enumerate(lv12.iterrows())][qi] if False else qi]
    nn_row = lv12.iloc[nn_id]
    print(f"  query: {lv12.iloc[qi]['title'][:30]:30s}  rating={y[qi]:.2f}"
          f"  →  NN: {lv12.iloc[nn_id]['title'][:30]:30s}  rating={y[nn_id]:.2f}"
          f"  Δ={abs(y[qi]-y[nn_id]):.2f}")

# ── 4. Reconstruction precision/recall on a sample ───────────────────────────
print("\nReconstruction precision/recall on 200 random windows (threshold=0.5):")
sample_charts = lv12.sample(20, random_state=0)
all_prec, all_rec = [], []

full_model = MAEModel(in_channels=IN_CH, num_patches=NUM_PATCHES_V4, patch_rows=PATCH_ROWS_V4,
                     encoder_dim=256, decoder_dim=128, num_heads=8,
                     use_lane_type_embed=True, use_bpm_cond=True)
full_model.encoder.load_state_dict(torch.load(CKPT, map_location='cpu'))
full_model = full_model.to(DEVICE).eval()

with torch.no_grad():
    for _, row in sample_charts.iterrows():
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        windows, _ = window_chart(enc, nc, WINDOW_BARS, STRIDE_BARS)
        if windows.shape[0] == 0:
            continue
        x   = torch.from_numpy(windows[:, :, :IN_CH]).float().to(DEVICE)
        bpm = torch.from_numpy(windows[:, :, 16].mean(axis=1)).float().to(DEVICE)
        B = x.shape[0]
        keep_ids, mask_ids = full_model._random_mask(B, DEVICE)
        encoded = full_model.encoder(x, keep_ids, bpm=bpm)
        pred    = full_model.decoder(encoded, keep_ids, mask_ids)

        target = x.reshape(B, NUM_PATCHES_V4, PATCH_ROWS_V4 * IN_CH)
        pred_m   = pred[torch.arange(B, device=DEVICE).unsqueeze(1), mask_ids]
        target_m = target[torch.arange(B, device=DEVICE).unsqueeze(1), mask_ids]

        pred_bin = (pred_m > 0.5).float()
        tgt_bin  = (target_m > 0.5).float()

        tp = (pred_bin * tgt_bin).sum().item()
        fp = (pred_bin * (1 - tgt_bin)).sum().item()
        fn = ((1 - pred_bin) * tgt_bin).sum().item()

        if tp + fp > 0:
            all_prec.append(tp / (tp + fp))
        if tp + fn > 0:
            all_rec.append(tp / (tp + fn))

print(f"  precision on note cells: {np.mean(all_prec):.3f}")
print(f"  recall    on note cells: {np.mean(all_rec):.3f}")
print(f"  F1:                      {2*np.mean(all_prec)*np.mean(all_rec)/(np.mean(all_prec)+np.mean(all_rec)):.3f}")
