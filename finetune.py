"""
Gauge-simulation fine-tuning:
  frozen v6 encoder → damage head → differentiable HC / EXH / EC simulation
  → trajectory features → rating heads (EC / HC / EXH)
  Evaluation: 5-fold CV MAE + Spearman ρ vs XGB baseline
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from tqdm import tqdm
from typing import List

torch.set_num_threads(1)  # sequential scalar sim: single thread faster than pool overhead

from dataset import (
    encode_chart_pretrain_v6, window_chart,
    WINDOW_BARS, STRIDE_BARS, PRETRAIN_IN_CHANNELS_V6,
    NUM_PATCHES_V4, PATCH_ROWS_V4,
)
from mae import MAEModel

# ── Config ────────────────────────────────────────────────────────────────────
MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
CKPT      = Path('checkpoints/pretrain_v6/encoder_best.pt')
ENC_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE     = torch.device('cpu')   # GaugeSimModel on CPU: sequential scalar ops are ~100x faster vs CUDA
TARGETS   = ['rating_ec', 'rating_hc', 'rating_exh']

EMBED_DIM    = 256
LR           = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS       = 100

XGB_MAE = {'rating_ec': 1.4570, 'rating_hc': 1.0998, 'rating_exh': 0.8363}
XGB_RHO = {'rating_ec': 0.7303, 'rating_hc': 0.7316, 'rating_exh': 0.6920}


# ── Encoder ───────────────────────────────────────────────────────────────────

def load_encoder():
    model = MAEModel(
        in_channels=PRETRAIN_IN_CHANNELS_V6,
        num_patches=NUM_PATCHES_V4, patch_rows=PATCH_ROWS_V4,
        encoder_dim=EMBED_DIM, decoder_dim=128, num_heads=8,
        use_lane_type_embed=True, use_bpm_cond=True,
    )
    model.encoder.load_state_dict(torch.load(CKPT, map_location=ENC_DEVICE))
    encoder = model.encoder.eval().to(ENC_DEVICE)
    for p in encoder.parameters():
        p.requires_grad_(False)
    return encoder


# ── Pre-compute embeddings + note counts ──────────────────────────────────────

def precompute(encoder, manifest):
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        manifest['rating_hc'].notna() &
        manifest['rating_exh'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    results, skipped = [], 0
    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='precomputing'):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            skipped += 1
            continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        windows, win_nc = window_chart(enc, nc, WINDOW_BARS, STRIDE_BARS)
        if windows.shape[0] == 0:
            skipped += 1
            continue
        x   = torch.from_numpy(windows[:, :, :16]).float().to(ENC_DEVICE)
        bpm = torch.from_numpy(windows[:, :, 16].mean(axis=1)).float().to(ENC_DEVICE)
        with torch.no_grad():
            cls = encoder.embed_segment(x, bpm=bpm)  # (T, 256)
        results.append({
            'emb':         cls.cpu(),
            'note_counts': torch.from_numpy(win_nc.astype(np.float32)),  # (T,)
            'total_notes': int(win_nc.sum()),
            'rating_ec':   float(row['rating_ec']),
            'rating_hc':   float(row['rating_hc']),
            'rating_exh':  float(row['rating_exh']),
        })
    print(f'precomputed {len(results)} charts  (skipped {skipped})', flush=True)
    return results


# ── Differentiable gauge simulations ──────────────────────────────────────────

@torch.jit.script
def simulate_hc(r: torch.Tensor, d_bad: torch.Tensor, d_poor: torch.Tensor, nc: torch.Tensor) -> torch.Tensor:
    """HC survival gauge: 30% correction, starts full."""
    g = torch.ones(1, dtype=r.dtype)
    traj: List[torch.Tensor] = []
    for t in range(r.shape[0]):
        recovery   = r[t]     * nc[t] * 0.0016
        drain_bad  = d_bad[t] * nc[t] * 0.05
        drain_poor = d_poor[t] * nc[t] * 0.09
        corr = torch.sigmoid((g - 0.30) * 20.0)
        drain_bad  = drain_bad * (0.5 + 0.5 * corr)
        g = torch.clamp(g + recovery - drain_bad - drain_poor, 0.0, 1.0)
        traj.append(g)
    return torch.cat(traj)


@torch.jit.script
def simulate_exh(r: torch.Tensor, d_bad: torch.Tensor, d_poor: torch.Tensor, nc: torch.Tensor) -> torch.Tensor:
    """EXH: no 30% correction, 2× damage multipliers."""
    g = torch.ones(1, dtype=r.dtype)
    traj: List[torch.Tensor] = []
    for t in range(r.shape[0]):
        recovery   = r[t]     * nc[t] * 0.0016
        drain_bad  = d_bad[t] * nc[t] * 0.10
        drain_poor = d_poor[t] * nc[t] * 0.18
        g = torch.clamp(g + recovery - drain_bad - drain_poor, 0.0, 1.0)
        traj.append(g)
    return torch.cat(traj)


@torch.jit.script
def simulate_ec(r: torch.Tensor, d_bad: torch.Tensor, d_poor: torch.Tensor, nc: torch.Tensor, total_notes: int) -> torch.Tensor:
    """EC groove gauge: starts at 20%, recovery scales with chart length."""
    a = 0.8 / max(float(total_notes), 350.0)
    g = torch.full((1,), 0.2, dtype=r.dtype)
    traj: List[torch.Tensor] = []
    for t in range(r.shape[0]):
        recovery = r[t]     * nc[t] * a
        drain    = d_bad[t] * nc[t] * 0.048
        g = torch.clamp(g + recovery - drain, 0.0, 1.0)
        traj.append(g)
    return torch.cat(traj)


def traj_features(traj):
    """5-dim trajectory summary: [min, final, mean, std, frac_below_30pct]."""
    std = traj.std() if traj.shape[0] > 1 else traj.new_zeros(1).squeeze()
    return torch.stack([
        traj.min(),
        traj[-1],
        traj.mean(),
        std,
        (traj < 0.30).float().mean(),
    ])


# ── Model ─────────────────────────────────────────────────────────────────────

class GaugeSimModel(nn.Module):
    """damage_head → 3 gauge simulators → trajectory features → 3 rating heads."""

    def __init__(self, embed_dim=EMBED_DIM):
        super().__init__()
        self.damage_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 3),
            nn.Sigmoid(),
        )
        self.rating_heads = nn.ModuleList([
            nn.Linear(5, 1) for _ in range(3)   # ec, hc, exh
        ])
        self._init_weights()

    def _init_weights(self):
        # bias damage outputs toward physically plausible starting values:
        # r≈0.7, d_bad≈0.05, d_poor≈0.02  →  logit values
        with torch.no_grad():
            self.damage_head[3].bias.copy_(
                torch.tensor([0.85, -2.94, -3.89])
            )
        # bias rating heads to approximate mean label (~12.0)
        for head in self.rating_heads:
            nn.init.constant_(head.bias, 12.0)

    def forward(self, emb, note_counts, total_notes):
        # emb: (T, 256),  note_counts: (T,)
        dmg = self.damage_head(emb)           # (T, 3)
        r, d_bad, d_poor = dmg[:, 0], dmg[:, 1], dmg[:, 2]
        nc  = note_counts.to(emb.device)

        hc_feat  = traj_features(simulate_hc(r, d_bad, d_poor, nc))
        exh_feat = traj_features(simulate_exh(r, d_bad, d_poor, nc))
        ec_feat  = traj_features(simulate_ec(r, d_bad, d_poor, nc, total_notes))

        return torch.stack([
            self.rating_heads[0](ec_feat).squeeze(),
            self.rating_heads[1](hc_feat).squeeze(),
            self.rating_heads[2](exh_feat).squeeze(),
        ])                                     # (3,)


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(train_data, val_data, epochs=EPOCHS):
    model = GaugeSimModel().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng   = np.random.default_rng(42)

    best_val, best_state = float('inf'), None

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for i in rng.permutation(len(train_data)):
            item = train_data[i]
            emb  = item['emb']              # already CPU tensor
            tgt  = torch.tensor(
                [item['rating_ec'], item['rating_hc'], item['rating_exh']],
                dtype=torch.float32,
            )
            pred = model(emb, item['note_counts'], item['total_notes'])
            loss = F.mse_loss(pred, tgt)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for item in val_data:
                tgt = torch.tensor(
                    [item['rating_ec'], item['rating_hc'], item['rating_exh']],
                    dtype=torch.float32,
                )
                val_loss += F.mse_loss(
                    model(item['emb'], item['note_counts'], item['total_notes']), tgt
                ).item()

        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0:
            print(f'    ep {epoch:3d}  train={total_loss/len(train_data):.4f}'
                  f'  val={val_loss/len(val_data):.4f}', flush=True)

    model.load_state_dict(best_state)
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, data):
    model.eval()
    preds = {t: [] for t in TARGETS}
    trues = {t: [] for t in TARGETS}
    with torch.no_grad():
        for item in data:
            pred = model(
                item['emb'], item['note_counts'], item['total_notes']
            ).numpy()
            for i, t in enumerate(TARGETS):
                preds[t].append(float(pred[i]))
                trues[t].append(item[t])
    return preds, trues


# ── Main ──────────────────────────────────────────────────────────────────────

manifest = pd.read_csv(MANIFEST)
encoder  = load_encoder()
print(f'encoder loaded from {CKPT}  (frozen, {ENC_DEVICE})', flush=True)

data = precompute(encoder, manifest)

total_params = sum(p.numel() for p in GaugeSimModel().parameters())
print(f'gauge sim model params: {total_params:,}', flush=True)
print(f'running 5-fold CV  ({EPOCHS} epochs/fold) on {DEVICE} ...', flush=True)

kf        = KFold(n_splits=5, shuffle=True, random_state=42)
fold_maes = {t: [] for t in TARGETS}
fold_rhos = {t: [] for t in TARGETS}

for fold, (train_idx, val_idx) in enumerate(kf.split(data), 1):
    train_data = [data[i] for i in train_idx]
    val_data   = [data[i] for i in val_idx]
    print(f'\n── fold {fold} ──', flush=True)
    model = train_fold(train_data, val_data)
    preds, trues = evaluate(model, val_data)

    parts = [f'fold {fold}:']
    for t in TARGETS:
        p   = np.array(preds[t])
        y   = np.array(trues[t])
        mae = float(np.abs(p - y).mean())
        rho = float(spearmanr(y, p).statistic)
        fold_maes[t].append(mae)
        fold_rhos[t].append(rho)
        parts.append(f'  {t.split("_")[1].upper()} MAE={mae:.4f} ρ={rho:.3f}')
    print(''.join(parts), flush=True)

print('\n── Results ──────────────────────────────────────────────────────────────', flush=True)
print(f'{"target":<12}  {"naive":>7}  {"XGB":>7}  {"gauge-sim":>11}  {"XGB ρ":>7}  {"gs ρ":>7}')
print('─' * 68)
for t in TARGETS:
    all_labels = np.array([d[t] for d in data])
    naive = float(np.abs(all_labels - all_labels.mean()).mean())
    mae   = np.mean(fold_maes[t])
    mae_s = np.std(fold_maes[t])
    rho   = np.mean(fold_rhos[t])
    print(f'{t:<12}  {naive:7.4f}  {XGB_MAE[t]:7.4f}  '
          f'{mae:6.4f}±{mae_s:.4f}  {XGB_RHO[t]:7.4f}  {rho:7.4f}', flush=True)
