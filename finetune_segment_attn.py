"""
Frozen half-chart encoder with learned segment attention pooling.

Each chart is divided into N equal temporal segments; each segment's
P1+P2 embeddings are fused as [mean, diff] → 512-dim.
A tiny model learns per-position weights (softmax attention) to pool
the N segment embeddings into a single chart representation, then
predicts EC/HC/EXH ratings via a small MLP head.

~33K parameters trained per fold with early stopping.
Compare against best fixed-weight result (k=5, ρ≈0.77).
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from scipy.stats import spearmanr
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
MANIFEST   = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT  = Path('/home/jysuh/projects/iidx_data')
CKPT       = Path('checkpoints/pretrain_half/encoder_best.pt')
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TARGETS    = ['rating_ec', 'rating_hc', 'rating_exh']
EMBED_DIM  = 256
SEG_DIM    = EMBED_DIM * 2   # 512 after [mean, diff] fusion
NOTE_DIM   = 8
N_SEGS     = 10
EPOCHS     = 300
PATIENCE   = 30
LR         = 1e-3
WD         = 1e-2

BEST_RHO  = {'rating_ec': 0.7722, 'rating_hc': 0.7747, 'rating_exh': 0.7376}  # k=5/3
BEST_MAE  = {'rating_ec': 1.3888, 'rating_hc': 1.0200, 'rating_exh': 0.7780}
XGB_RHO   = {'rating_ec': 0.7303, 'rating_hc': 0.7316, 'rating_exh': 0.6920}


# ── Model ─────────────────────────────────────────────────────────────────────

class SegmentRater(nn.Module):
    """Learned softmax attention over N segment embeddings → MLP → 3 ratings."""
    def __init__(self, n_segs=N_SEGS, seg_dim=SEG_DIM, note_dim=NOTE_DIM):
        super().__init__()
        self.pos_w = nn.Parameter(torch.zeros(n_segs))   # position importance
        self.head = nn.Sequential(
            nn.Linear(seg_dim + note_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, segs, note_feats):
        # segs: (B, N, seg_dim),  note_feats: (B, note_dim)
        w      = torch.softmax(self.pos_w, dim=0)           # (N,)
        pooled = (segs * w[None, :, None]).sum(1)            # (B, seg_dim)
        return self.head(torch.cat([pooled, note_feats], -1))  # (B, 3)


# ── Encoder loading ───────────────────────────────────────────────────────────

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


# ── Pre-compute segment embeddings ────────────────────────────────────────────

@torch.no_grad()
def embed_windows(encoder, wins):
    x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
    bpm = torch.from_numpy(wins[:, :, 8].mean(axis=1)).float().to(DEVICE)
    return encoder.embed_segment(x, bpm=bpm).cpu().numpy()  # (T, 256)


def fused_segment_means(p1_cls, p2_cls, n_segs=N_SEGS):
    """Split T windows into n_segs equal parts; return (n_segs, 512) fused means."""
    p1_parts = np.array_split(p1_cls, n_segs)
    p2_parts = np.array_split(p2_cls, n_segs)
    out = []
    for p1, p2 in zip(p1_parts, p2_parts):
        m1, m2 = p1.mean(0), p2.mean(0)
        out.append(np.concatenate([(m1 + m2) / 2.0, m1 - m2]))  # (512,)
    return np.stack(out)  # (n_segs, 512)


def precompute(encoder, manifest):
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        manifest['rating_hc'].notna() &
        manifest['rating_exh'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    all_segs, note_feats, labels = [], [], []
    skipped = 0
    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='embedding charts'):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            skipped += 1; continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        p1_wins, p2_wins, win_nc = window_chart_split(enc, nc, WINDOW_BARS, STRIDE_BARS)
        if p1_wins.shape[0] == 0:
            skipped += 1; continue

        p1_cls = embed_windows(encoder, p1_wins)
        p2_cls = embed_windows(encoder, p2_wins)
        all_segs.append(fused_segment_means(p1_cls, p2_cls))  # (N, 512)

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

    print(f'embedded {len(all_segs)} charts  (skipped {skipped})', flush=True)
    # all_segs: list of (N, 512) arrays
    return np.stack(all_segs), np.array(note_feats), np.array(labels)


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(segs_tr, note_tr, y_tr, segs_va, note_va, y_va,
               seg_scaler, note_scaler):
    """Train one fold; return val predictions."""

    # standardise using training stats
    B_tr, N, D = segs_tr.shape
    segs_tr_s = seg_scaler.transform(segs_tr.reshape(-1, D)).reshape(B_tr, N, D)
    segs_va_s = seg_scaler.transform(segs_va.reshape(-1, D)).reshape(-1, N, D)
    note_tr_s = note_scaler.transform(note_tr)
    note_va_s = note_scaler.transform(note_va)

    # normalise targets by training std
    y_std = torch.tensor(y_tr.std(0), dtype=torch.float32, device=DEVICE).clamp(min=0.1)

    X_tr  = torch.tensor(segs_tr_s,  dtype=torch.float32, device=DEVICE)
    nf_tr = torch.tensor(note_tr_s,  dtype=torch.float32, device=DEVICE)
    y_tr_ = torch.tensor(y_tr,       dtype=torch.float32, device=DEVICE)
    X_va  = torch.tensor(segs_va_s,  dtype=torch.float32, device=DEVICE)
    nf_va = torch.tensor(note_va_s,  dtype=torch.float32, device=DEVICE)
    y_va_ = torch.tensor(y_va,       dtype=torch.float32, device=DEVICE)

    model = SegmentRater().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_val, best_preds, patience_ctr = float('inf'), None, 0
    rng = np.random.default_rng(42)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = rng.permutation(len(X_tr))
        for i in idx:
            pred = model(X_tr[i:i+1], nf_tr[i:i+1])
            loss = ((pred - y_tr_[i:i+1]) / y_std).pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            pred_va = model(X_va, nf_va)
            val_loss = ((pred_va - y_va_) / y_std).pow(2).mean().item()

        if val_loss < best_val:
            best_val = val_loss
            best_preds = pred_va.cpu().numpy()
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= PATIENCE:
            break

    return best_preds   # (B_va, 3)


# ── Main ──────────────────────────────────────────────────────────────────────

print(f'device: {DEVICE}', flush=True)
encoder  = load_encoder()
manifest = pd.read_csv(MANIFEST)
all_segs, X_note, y = precompute(encoder, manifest)
# all_segs: (N_charts, N_SEGS, SEG_DIM)
print(f'segments: {all_segs.shape}  note: {X_note.shape}  y: {y.shape}', flush=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_maes = {t: [] for t in TARGETS}
fold_rhos = {t: [] for t in TARGETS}

for fold, (tr_idx, va_idx) in enumerate(kf.split(all_segs), 1):
    segs_tr, segs_va = all_segs[tr_idx], all_segs[va_idx]
    note_tr, note_va = X_note[tr_idx],   X_note[va_idx]
    y_tr,    y_va    = y[tr_idx],         y[va_idx]

    # fit scalers on training fold only
    B_tr, N, D = segs_tr.shape
    seg_scaler  = StandardScaler().fit(segs_tr.reshape(-1, D))
    note_scaler = StandardScaler().fit(note_tr)

    preds = train_fold(segs_tr, note_tr, y_tr, segs_va, note_va, y_va,
                       seg_scaler, note_scaler)

    parts = [f'fold {fold}:']
    for j, t in enumerate(TARGETS):
        mae = float(np.abs(preds[:, j] - y_va[:, j]).mean())
        rho = float(spearmanr(y_va[:, j], preds[:, j]).statistic)
        fold_maes[t].append(mae)
        fold_rhos[t].append(rho)
        parts.append(f'  {t.split("_")[1].upper()} MAE={mae:.4f} ρ={rho:.3f}')
    print(''.join(parts), flush=True)

print('\n── Results ─────────────────────────────────────────────────────────────', flush=True)
print(f'{"target":<12}  {"XGB ρ":>7}  {"best-k ρ":>9}  {"seg-attn ρ":>11}  '
      f'{"best-k MAE":>11}  {"seg-attn MAE":>13}')
print('─' * 75)
for t in TARGETS:
    rho = np.mean(fold_rhos[t])
    mae = np.mean(fold_maes[t])
    print(f'{t:<12}  {XGB_RHO[t]:7.4f}  {BEST_RHO[t]:9.4f}  {rho:11.4f}  '
          f'{BEST_MAE[t]:11.4f}  {mae:13.4f}', flush=True)
