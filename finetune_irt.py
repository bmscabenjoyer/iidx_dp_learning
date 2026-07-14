"""
IRT-based gauge simulation fine-tuning.

Per-window chart difficulty d[t] is predicted by the encoder.
For a hypothetical player with skill θ, hit rates follow IRT:
  recovery_rate(θ, d[t]) = sigmoid(α * (θ - d[t]))
  drain_rate(θ, d[t])    = sigmoid(α * (d[t] - θ))   (= 1 − recovery)

Gauge trajectories are simulated in parallel for N_theta skill levels.
The predicted rating θ* is the θ at which pass probability crosses 0.5
(differentiably extracted as a weighted mean over the transition region).

EC uses final gauge ≥ 0.80.  HC/EXH use min gauge > 0 (survival).
Separate learned physics scales (recovery/drain magnitudes) per gauge type
let EC, HC, EXH naturally produce different θ* from the same d[t].

5-fold CV on 684 rated lv12 charts.  Encoder unfrozen (lr=1e-5).
Reuses cache/e2e_v11/ from finetune_e2e.py.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import (
    PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
    NUM_PATCHES_V10, PATCH_ROWS_V7,
)
from mae import MAEModel

MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
CKPT      = Path('checkpoints/pretrain_half_v11/encoder_best.pt')
CACHE_DIR = Path('cache/e2e_v11')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 192

THETA_MIN = 0.0
THETA_MAX = 15.0
N_THETA   = 150


# ── encoder ──────────────────────────────────────────────────────────────────

def load_encoder():
    model = MAEModel(
        in_channels=PRETRAIN_IN_CHANNELS_HALF,
        num_patches=NUM_PATCHES_V10, patch_rows=PATCH_ROWS_V7,
        encoder_dim=EMBED_DIM, decoder_dim=96, num_heads=4,
        encoder_depth=4, use_rope=True,
        use_lane_type_embed=True, use_bpm_cond=False,
        lane_types=LANE_TYPES_HALF,
    )
    model.encoder.load_state_dict(torch.load(CKPT, map_location='cpu'))
    return model.encoder


# ── dataset ───────────────────────────────────────────────────────────────────

class ChartDataset(Dataset):
    def __init__(self, lv12, indices, augment=True):
        self.samples = []
        for i in indices:
            p = CACHE_DIR / f'{i}.npz'
            if not p.exists():
                continue
            row    = lv12.iloc[i]
            data   = np.load(p)
            p1     = data['p1'].astype(np.float32)    # (T, WIN_ROWS, 8)
            p2     = data['p2'].astype(np.float32)
            win_nc = data['win_nc'].astype(np.float32) # (T,)
            ec     = float(row['rating_ec'])
            hc     = float(row['rating_hc'])  if pd.notna(row['rating_hc'])  else ec
            exh    = float(row['rating_exh']) if pd.notna(row['rating_exh']) else ec
            y      = np.array([ec, hc, exh], dtype=np.float32)
            self.samples.append((p1, p2, win_nc, y, i))
            if augment:
                self.samples.append((p2, p1, win_nc, y, i))  # flip: swap sides

    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate(batch):
    p1s, p2s, ncs, ys, idxs = zip(*batch)
    lengths = [p.shape[0] for p in p1s]
    T_max   = max(lengths)
    B, R, C = len(p1s), p1s[0].shape[1], p1s[0].shape[2]

    padded_p1 = torch.zeros(B, T_max, R, C)
    padded_p2 = torch.zeros(B, T_max, R, C)
    padded_nc = torch.zeros(B, T_max)
    mask      = torch.ones(B, T_max, dtype=torch.bool)  # True = padded

    for i, (p1, p2, nc, l) in enumerate(zip(p1s, p2s, ncs, lengths)):
        padded_p1[i, :l] = torch.from_numpy(p1)
        padded_p2[i, :l] = torch.from_numpy(p2)
        padded_nc[i, :l] = torch.from_numpy(nc)
        mask[i, :l]      = False

    return (padded_p1, padded_p2, padded_nc, mask,
            torch.from_numpy(np.stack(ys)),
            list(idxs))


# ── model ─────────────────────────────────────────────────────────────────────

class IRTGaugeModel(nn.Module):
    """
    Encoder → d[t] per window → IRT gauge simulation over θ grid → θ* rating.

    θ* = minimum player skill that can pass the chart on each gauge type.
    The θ scale matches the ereter rating scale (roughly 0–15).
    """

    def __init__(self, encoder, n_theta=N_THETA, theta_min=THETA_MIN, theta_max=THETA_MAX):
        super().__init__()
        self.encoder   = encoder
        self.n_theta   = n_theta
        self.theta_min = theta_min
        self.theta_max = theta_max

        in_dim = EMBED_DIM * 2 + 1  # [p1 ∥ p2 ∥ position]

        # Separate per-gauge difficulty heads with gauge-aligned bias init.
        # EC and HC/EXH have fundamentally different pass conditions:
        #   EC must ACCUMULATE to 80% → θ* ≈ d_ec[t] (borderline at skill=diff)
        #   HC/EXH must SURVIVE (not hit 0) → θ* ≈ d_hc[t] similarly
        # Separate heads let each gauge operate in its own rating range,
        # avoiding the ordering flip that occurs when a single d[t] must
        # simultaneously satisfy incompatible gauge physics.
        def _make_head(bias):
            head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, 64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 1),
            )
            nn.init.constant_(head[-1].bias, bias)
            return head

        # Biases initialised to ereter means: EC≈5.3, HC≈8.4, EXH≈11.3
        self.diff_ec  = _make_head(5.3)
        self.diff_hc  = _make_head(8.4)
        self.diff_exh = _make_head(11.3)

        # IRT sharpness α (shared) — kept moderate so skill gaps matter
        self.log_sharpness = nn.Parameter(torch.tensor(0.0))  # softplus(0)+0.5 ≈ 1.19

        # Gauge physics scales (log-parameterised, positive).
        # With d_g[t] centred on each gauge's mean, at θ=d_g[t] (r=0.5):
        #   EC needs net delta ≥ 0.58 over chart → rec − drain ≥ 1.16
        #   HC needs net delta ≥ −1.0 over chart → drain − rec ≤ 1.0  (just survive)
        #   EXH same as HC but drain higher
        # Physics calibrated so each gauge is borderline at θ=d_g[t] when r=0.5:
        #   EC: start=0.22, need final≥0.80 → net delta=0.58, so rec-drain=1.16
        #       softplus(1.0)=1.313, softplus(-1.8)=0.153 → rec-drain=1.16 ✓
        #   HC: survival, start=1.0, need min>0 → drain-rec≤2.0 per note-weighted unit
        #       softplus(0.0)=0.693, softplus(2.7)=2.730 → drain-rec=2.04 ✓
        #   EXH: same condition, same init; harsher drain in reality → training will diverge
        self.log_ec_rec    = nn.Parameter(torch.tensor( 1.0))   # softplus ≈ 1.313
        self.log_ec_drain  = nn.Parameter(torch.tensor(-1.8))   # softplus ≈ 0.153
        self.log_hc_rec    = nn.Parameter(torch.tensor( 0.0))   # softplus ≈ 0.693
        self.log_hc_drain  = nn.Parameter(torch.tensor( 2.7))   # softplus ≈ 2.730
        self.log_exh_drain = nn.Parameter(torch.tensor( 2.7))   # softplus ≈ 2.730

    # ── differentiable rating extraction ────────────────────────────────────

    def _extract_rating(self, pass_prob, theta):
        """
        pass_prob: (B, N_theta) — soft pass probability, should be ↑ in θ
        theta:     (N_theta,)
        Returns:   (B,) — θ at the 0→1 transition (expected value of derivative)
        """
        dpass     = (pass_prob[:, 1:] - pass_prob[:, :-1]).clamp(min=0.0)  # (B, N-1)
        theta_mid = (theta[1:] + theta[:-1]) / 2.0                          # (N-1,)
        total     = dpass.sum(1, keepdim=True).clamp(min=1e-6)
        return (dpass * theta_mid.unsqueeze(0)).sum(1) / total.squeeze(1)   # (B,)

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, p1, p2, win_nc, mask):
        """
        p1, p2:  (B, T, WIN_ROWS, 8)
        win_nc:  (B, T)  note counts per window
        mask:    (B, T)  True = padded
        Returns: (B, 3)  [ec_rating, hc_rating, exh_rating]
        """
        B, T_max, R, C = p1.shape
        device = p1.device

        # Encode both sides
        e1 = self.encoder.embed_segment(p1.view(B * T_max, R, C)).view(B, T_max, EMBED_DIM)
        e2 = self.encoder.embed_segment(p2.view(B * T_max, R, C)).view(B, T_max, EMBED_DIM)

        if mask is not None:
            m  = mask.unsqueeze(-1)
            e1 = e1.masked_fill(m, 0.0)
            e2 = e2.masked_fill(m, 0.0)

        pos = torch.linspace(0, 1, T_max, device=device).view(1, T_max, 1).expand(B, -1, -1)
        x   = torch.cat([e1, e2, pos], dim=-1)              # (B, T, 385)

        # Per-window chart difficulty from gauge-specific heads
        d_ec  = self.diff_ec(x).squeeze(-1)                 # (B, T)
        d_hc  = self.diff_hc(x).squeeze(-1)
        d_exh = self.diff_exh(x).squeeze(-1)
        if mask is not None:
            d_ec  = d_ec.masked_fill(mask,  0.0)
            d_hc  = d_hc.masked_fill(mask,  0.0)
            d_exh = d_exh.masked_fill(mask, 0.0)

        # θ grid: (N_theta,)
        theta = torch.linspace(self.theta_min, self.theta_max, self.n_theta, device=device)
        alpha = F.softplus(self.log_sharpness) + 0.5        # shared sharpness, ≥ 0.5

        # Note density weighting: (B, T, 1)
        nc     = win_nc.unsqueeze(-1)
        nc_rel = nc / nc.sum(1, keepdim=True).clamp(min=1)
        if mask is not None:
            nc_rel = nc_rel.masked_fill(mask.unsqueeze(-1), 0.0)

        # Physics scales (positive)
        ec_rec    = F.softplus(self.log_ec_rec)
        ec_drain  = F.softplus(self.log_ec_drain)
        hc_rec    = F.softplus(self.log_hc_rec)
        hc_drain  = F.softplus(self.log_hc_drain)
        exh_drain = F.softplus(self.log_exh_drain)

        def _irt_rates(d):
            gap = theta.view(1, 1, -1) - d.unsqueeze(-1)    # (B, T, N)
            r   = torch.sigmoid(alpha * gap)
            b   = torch.sigmoid(-alpha * gap)
            return r, b

        # ── EC gauge: accumulates, must end ≥ 80% ───────────────────────────
        r_ec, b_ec   = _irt_rates(d_ec)
        ec_delta     = (r_ec * ec_rec - b_ec * ec_drain) * nc_rel  # (B, T, N)
        ec_final     = 0.22 + ec_delta.sum(1)                       # (B, N)
        ec_pass      = torch.sigmoid(20.0 * (ec_final - 0.80))

        # ── HC gauge: survival, must never reach 0 ───────────────────────────
        r_hc, b_hc   = _irt_rates(d_hc)
        hc_delta     = (r_hc * hc_rec - b_hc * hc_drain) * nc_rel
        hc_traj      = 1.0 + hc_delta.cumsum(1)                     # (B, T, N)
        if mask is not None:
            hc_traj  = hc_traj.masked_fill(mask.unsqueeze(-1), 1e4)
        hc_min       = -torch.logsumexp(-hc_traj * 20.0, dim=1) / 20.0
        hc_pass      = torch.sigmoid(20.0 * hc_min)

        # ── EXH gauge: survival, harsher drain ──────────────────────────────
        r_exh, b_exh = _irt_rates(d_exh)
        exh_delta    = (r_exh * hc_rec - b_exh * exh_drain) * nc_rel
        exh_traj     = 1.0 + exh_delta.cumsum(1)
        if mask is not None:
            exh_traj = exh_traj.masked_fill(mask.unsqueeze(-1), 1e4)
        exh_min      = -torch.logsumexp(-exh_traj * 20.0, dim=1) / 20.0
        exh_pass     = torch.sigmoid(20.0 * exh_min)

        # θ* = θ at which pass_prob transitions 0 → 1
        ec_rating  = self._extract_rating(ec_pass,  theta)  # (B,)
        hc_rating  = self._extract_rating(hc_pass,  theta)
        exh_rating = self._extract_rating(exh_pass, theta)

        return torch.stack([ec_rating, hc_rating, exh_rating], dim=1)  # (B, 3)


# ── training ──────────────────────────────────────────────────────────────────

def run_fold(lv12, train_idx, val_idx, args):
    train_dl = DataLoader(ChartDataset(lv12, train_idx, augment=True),
                          batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(ChartDataset(lv12, val_idx, augment=False),
                          batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0)

    encoder = load_encoder().to(DEVICE)
    model   = IRTGaugeModel(encoder).to(DEVICE)

    head_params = (list(model.diff_ec.parameters()) +
                   list(model.diff_hc.parameters()) +
                   list(model.diff_exh.parameters()) +
                   [model.log_sharpness,
                    model.log_ec_rec,   model.log_ec_drain,
                    model.log_hc_rec,   model.log_hc_drain,
                    model.log_exh_drain])
    opt = optim.AdamW([
        {'params': encoder.parameters(), 'lr': args.encoder_lr},
        {'params': head_params,          'lr': args.head_lr},
    ], weight_decay=args.wd)

    warmup = args.warmup_epochs

    def enc_lr_lambda(ep):
        if ep < warmup:
            return ep / max(warmup, 1)
        t = (ep - warmup) / max(args.epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * t))

    def head_lr_lambda(ep):
        t = ep / max(args.epochs, 1)
        return 0.5 * (1 + np.cos(np.pi * t))

    sched    = optim.lr_scheduler.LambdaLR(opt, [enc_lr_lambda, head_lr_lambda])
    scaler   = torch.amp.GradScaler('cuda')
    use_amp  = DEVICE.type == 'cuda'
    best_val, best_state = float('inf'), None

    for epoch in tqdm(range(1, args.epochs + 1), desc='  training', leave=False):
        model.train()
        for p1, p2, nc, mask, y, _ in train_dl:
            p1, p2, nc, mask, y = (p1.to(DEVICE), p2.to(DEVICE), nc.to(DEVICE),
                                    mask.to(DEVICE), y.to(DEVICE))
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                pred = model(p1, p2, nc, mask)
                loss = F.mse_loss(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for p1, p2, nc, mask, y, _ in val_dl:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(p1.to(DEVICE), p2.to(DEVICE),
                                 nc.to(DEVICE), mask.to(DEVICE))
                val_loss += F.mse_loss(pred, y.to(DEVICE)).item()
        val_loss /= len(val_dl)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    preds, targets, chart_idxs = [], [], []
    with torch.no_grad():
        for p1, p2, nc, mask, y, idxs in val_dl:
            preds.append(model(p1.to(DEVICE), p2.to(DEVICE),
                               nc.to(DEVICE), mask.to(DEVICE)).cpu())
            targets.append(y)
            chart_idxs.extend(idxs)

    # Log learned physics parameters
    with torch.no_grad():
        alpha     = float(F.softplus(model.log_sharpness) + 0.5)
        ec_rec    = float(F.softplus(model.log_ec_rec))
        ec_drain  = float(F.softplus(model.log_ec_drain))
        hc_rec    = float(F.softplus(model.log_hc_rec))
        hc_drain  = float(F.softplus(model.log_hc_drain))
        exh_drain = float(F.softplus(model.log_exh_drain))
    print(f'  physics: α={alpha:.3f}  '
          f'ec(rec={ec_rec:.3f} drain={ec_drain:.3f})  '
          f'hc(rec={hc_rec:.3f} drain={hc_drain:.3f})  '
          f'exh_drain={exh_drain:.3f}', flush=True)

    return torch.cat(preds).numpy(), torch.cat(targets).numpy(), chart_idxs


# ── main ──────────────────────────────────────────────────────────────────────

def main(manifest_path=None, data_root=None, ckpt_path=None,
         cache_dir=None, results_dir=None):
    global MANIFEST, DATA_ROOT, CKPT, CACHE_DIR

    if manifest_path: MANIFEST  = Path(manifest_path)
    if data_root:     DATA_ROOT = Path(data_root)
    if ckpt_path:     CKPT      = Path(ckpt_path)
    if cache_dir:     CACHE_DIR = Path(cache_dir)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--batch-size',    type=int,   default=8)
    p.add_argument('--encoder-lr',    type=float, default=1e-5)
    p.add_argument('--head-lr',       type=float, default=1e-3)
    p.add_argument('--wd',            type=float, default=0.05)
    p.add_argument('--warmup-epochs', type=int,   default=5)
    p.add_argument('--folds',         type=int,   default=5)
    args = p.parse_args([])  # parse empty (called programmatically from Modal)

    print(f'device: {DEVICE}')
    manifest = pd.read_csv(MANIFEST)
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    valid_idx = [i for i in range(len(lv12)) if (CACHE_DIR / f'{i}.npz').exists()]
    print(f'charts with cache: {len(valid_idx)}')

    enc_params  = sum(p.numel() for p in load_encoder().parameters())
    irt_params  = sum(p.numel() for p in IRTGaugeModel(load_encoder()).parameters())
    irt_params -= enc_params
    print(f'encoder params: {enc_params:,}  IRT head params: {irt_params:,}', flush=True)

    out_dir = Path(results_dir or '.')
    out_dir.mkdir(parents=True, exist_ok=True)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_preds, all_targets, all_idxs = [], [], []

    for fold, (tr, va) in enumerate(kf.split(valid_idx), 1):
        train_idx = [valid_idx[i] for i in tr]
        val_idx   = [valid_idx[i] for i in va]
        print(f'\nfold {fold}/{args.folds}  train={len(train_idx)}  val={len(val_idx)}',
              flush=True)
        preds, targets, idxs = run_fold(lv12, train_idx, val_idx, args)
        # Save per-fold results
        fold_path = Path(results_dir or '.') / f'predictions_irt_fold{fold}.npz'
        np.savez(fold_path, preds=preds, targets=targets, idxs=np.array(idxs))
        all_preds.append(preds)
        all_targets.append(targets)
        all_idxs.extend(idxs)
        for j, name in enumerate(['ec', 'hc', 'exh']):
            mae = np.mean(np.abs(preds[:, j] - targets[:, j]))
            rho = spearmanr(preds[:, j], targets[:, j]).statistic
            print(f'  {name}: MAE={mae:.4f}  ρ={rho:.4f}', flush=True)

    print('\n── 5-fold aggregate ──────────────────────────────')
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    lines = []
    for j, name in enumerate(['ec', 'hc', 'exh']):
        mae = np.mean(np.abs(all_preds[:, j] - all_targets[:, j]))
        rho = spearmanr(all_preds[:, j], all_targets[:, j]).statistic
        line = f'  {name}: MAE={mae:.4f}  ρ={rho:.4f}'
        print(line)
        lines.append(line)

    # Save predictions CSV
    rows = []
    for k, i in enumerate(all_idxs):
        r = lv12.iloc[i]
        rows.append({
            'title':    r['title'],   'diftype':  r['diftype'],
            'ec':       all_targets[k, 0], 'hc': all_targets[k, 1], 'exh': all_targets[k, 2],
            'ec_pred':  all_preds[k, 0],   'hc_pred': all_preds[k, 1], 'exh_pred': all_preds[k, 2],
        })
    df = pd.DataFrame(rows)
    df['ec_err']  = df['ec_pred']  - df['ec']
    df['hc_err']  = df['hc_pred']  - df['hc']
    df['exh_err'] = df['exh_pred'] - df['exh']
    df.to_csv(out_dir / 'predictions_irt.csv', index=False)
    print(f'\npredictions → {out_dir}/predictions_irt.csv')

    # Save summary
    with open(out_dir / 'results_irt.txt', 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print('\n── EC outliers (|err| > 4) ──────────────────────')
    out = df[df['ec_err'].abs() > 4].sort_values('ec_err')
    for _, r in out.iterrows():
        print(f'  {r["title"]:<45} ec={r["ec"]:.1f}  pred={r["ec_pred"]:.1f}'
              f'  err={r["ec_err"]:+.1f}')


if __name__ == '__main__':
    main()
