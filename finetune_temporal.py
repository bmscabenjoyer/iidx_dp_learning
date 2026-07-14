"""
Gauge-simulation rating head on frozen v11 encoder.
Per-window P1+P2 embeddings (384-dim) + normalised position (1-dim)
→ per-window hardness scalar h[t] ∈ [0,1]
→ expected EC / HC / EXH gauge delta per window (physics, no learned params)
→ sequential gauge simulation
→ trajectory features → linear rating heads → [rating_ec, rating_hc, rating_exh]

EC: groove gauge, starts 22%, no instant-fail, must end ≥ 80%  → g_ec_final
HC: survival gauge, starts 100%, fails at 0%, 30% correction   → g_hc_min, g_hc_final
EXH: survival gauge, no 30% correction, harsher drain          → g_exh_min, g_exh_final

5-fold CV on 684 lv12 labeled charts.
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import (
    encode_chart_pretrain_v6, window_chart_time_half,
    PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
    NUM_PATCHES_V10, PATCH_ROWS_V7,
    WIN_SECS_V10, WIN_ROWS_V10,
    HEAD_TYPES,
)
from mae import MAEModel

MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
CKPT      = Path('checkpoints/pretrain_half_v11/encoder_best.pt')
CACHE_DIR = Path('cache/temporal_v11')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 192


# ── encoder ──────────────────────────────────────────────────────────────────

def load_encoder():
    model = MAEModel(
        in_channels=PRETRAIN_IN_CHANNELS_HALF,
        num_patches=NUM_PATCHES_V10, patch_rows=PATCH_ROWS_V7,
        encoder_dim=192, decoder_dim=96, num_heads=4,
        encoder_depth=4, use_rope=True,
        use_lane_type_embed=True, use_bpm_cond=False,
        lane_types=LANE_TYPES_HALF,
    )
    model.encoder.load_state_dict(torch.load(CKPT, map_location='cpu'))
    return model.encoder.eval().to(DEVICE)


@torch.no_grad()
def _embed(encoder, wins):
    x   = torch.from_numpy(wins[:, :, :8]).float().to(DEVICE)
    bpm = torch.from_numpy(wins[:, :, 8].mean(axis=1)).float().to(DEVICE)
    return encoder.embed_segment(x, bpm=bpm).cpu().numpy()


# ── embedding cache ───────────────────────────────────────────────────────────

def build_cache(manifest):
    """Encode all rated lv12 charts and cache embeddings, win_nc, total_notes."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    head_arr = np.array(sorted(HEAD_TYPES), dtype=np.uint32)
    missing = [i for i in range(len(lv12))
               if not (CACHE_DIR / f'{i}.npz').exists()
               or 'total_notes' not in np.load(CACHE_DIR / f'{i}.npz')]
    if not missing:
        print(f'cache complete ({len(lv12)} charts)')
        return lv12

    print(f'building cache: {len(missing)} charts to encode...')
    encoder = load_encoder()
    for i in tqdm(missing):
        row = lv12.iloc[i]
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        p1_wins, p2_wins, win_nc = window_chart_time_half(
            enc, nc, win_secs=WIN_SECS_V10, stride_secs=WIN_SECS_V10 / 2,
            win_rows=WIN_ROWS_V10,
        )
        if p1_wins.shape[0] == 0:
            continue
        embs = np.concatenate([_embed(encoder, p1_wins),
                                _embed(encoder, p2_wins)], axis=1).astype(np.float32)
        # total actionable notes across both P1 and P2 (used for EC recovery constant)
        raw_lanes = np.load(str(npy))[:, :16].astype(np.uint32)
        total_notes = int(np.isin(raw_lanes, head_arr).sum())
        np.savez(CACHE_DIR / f'{i}.npz',
                 embs=embs,
                 win_nc=win_nc.astype(np.float32),
                 total_notes=np.array(total_notes, dtype=np.int32))
    return lv12


# ── dataset ───────────────────────────────────────────────────────────────────

class ChartDataset(Dataset):
    def __init__(self, lv12, indices, augment=True):
        self.samples = []
        half = EMBED_DIM  # 192
        for i in indices:
            p = CACHE_DIR / f'{i}.npz'
            if not p.exists():
                continue
            row    = lv12.iloc[i]
            data   = np.load(p)
            embs   = data['embs']                          # (T, 384)
            win_nc = data['win_nc']                        # (T,)
            total_notes = int(data['total_notes'])
            T      = embs.shape[0]
            pos    = np.linspace(0, 1, T, dtype=np.float32).reshape(-1, 1)
            x      = np.concatenate([embs, pos], axis=1)  # (T, 385)
            ec   = float(row['rating_ec'])
            hc   = float(row['rating_hc'])  if pd.notna(row['rating_hc'])  else ec
            exh  = float(row['rating_exh']) if pd.notna(row['rating_exh']) else ec
            y    = np.array([ec, hc, exh], dtype=np.float32)
            self.samples.append((x, win_nc, total_notes, y, i))
            if augment:
                embs_flip = np.concatenate([embs[:, half:], embs[:, :half]], axis=1)
                x_flip    = np.concatenate([embs_flip, pos], axis=1)
                self.samples.append((x_flip, win_nc, total_notes, y, i))

    def __len__(self):         return len(self.samples)
    def __getitem__(self, i):  return self.samples[i]


def collate(batch):
    xs, ncs, nts, ys, idxs = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    T_max   = max(lengths)
    padded  = pad_sequence([torch.from_numpy(x) for x in xs], batch_first=True)
    padded_nc = torch.zeros(len(xs), T_max, dtype=torch.float32)
    mask      = torch.zeros(len(xs), T_max, dtype=torch.bool)
    for i, (nc, l) in enumerate(zip(ncs, lengths)):
        padded_nc[i, :l] = torch.from_numpy(nc)
        mask[i, l:] = True
    return (padded, padded_nc,
            torch.tensor(nts, dtype=torch.long),
            mask,
            torch.from_numpy(np.stack(ys)),
            list(idxs))


# ── model ─────────────────────────────────────────────────────────────────────

class GaugeSimHead(nn.Module):
    """
    Per-window difficulty score s[t] ∈ [-1, 1] from embedding + position.
    Gauge delta = s[t] × relative_density[t] × learned_scale (per gauge type).
    Vectorised simulation via cumsum — no Python loop, gradient-friendly.

    EC: no instant-fail → final value of running sum matters (order-invariant).
    HC: instant-fail at 0 → minimum of running sum matters + 30% correction approx.
    EXH: same as HC, larger (learned) scale.

    Key inductive bias: hard endings hurt HC/EXH more than EC because the
    minimum of the running sum captures the worst-ever gauge, while EC only
    cares about the final value (recoverable if chart ends easy).
    """
    def __init__(self, in_dim=385, dropout=0.2):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Tanh(),  # ∈ [-1, 1]: positive=recovery, negative=drain
        )
        # learned sensitivity per gauge type (EXH should be > HC > EC)
        self.ec_scale  = nn.Parameter(torch.tensor(0.1))
        self.hc_scale  = nn.Parameter(torch.tensor(0.1))
        self.exh_scale = nn.Parameter(torch.tensor(0.1))

        self.ec_head  = nn.Linear(1, 1)   # ec_final
        self.hc_head  = nn.Linear(2, 1)   # [hc_min, hc_final]
        self.exh_head = nn.Linear(2, 1)   # [exh_min, exh_final]

    def forward(self, x, win_nc, total_notes, mask=None):
        # x:           (B, T, 385)
        # win_nc:      (B, T)   note counts (0-padded)
        # total_notes: (B,)     total chart notes
        # mask:        (B, T)   True = padded
        B, T, _ = x.shape

        s = self.score(x).squeeze(-1)  # (B, T), ∈ [-1, 1]

        # weight by relative note density within each window
        nc_rel = win_nc / (win_nc.mean(1, keepdim=True).clamp(min=1))  # (B, T)
        if mask is not None:
            nc_rel = nc_rel.masked_fill(mask, 0.0)

        # gauge delta per gauge type (vectorised, no loop)
        ec_delta  = s * nc_rel * self.ec_scale
        hc_delta  = s * nc_rel * self.hc_scale
        exh_delta = s * nc_rel * self.exh_scale

        # EC: final running sum (no instant-fail, order-invariant)
        ec_final = 0.22 + ec_delta.sum(1)          # (B,)

        # HC / EXH: minimum of running sum (hard section anywhere is dangerous)
        hc_run  = 1.0 + hc_delta.cumsum(1)         # (B, T)
        exh_run = 1.0 + exh_delta.cumsum(1)        # (B, T)
        if mask is not None:
            hc_run  = hc_run.masked_fill(mask,  1e4)
            exh_run = exh_run.masked_fill(mask, 1e4)

        hc_min   = hc_run.min(1).values             # (B,)
        hc_final = 1.0 + hc_delta.sum(1)
        exh_min  = exh_run.min(1).values            # (B,)
        exh_final = 1.0 + exh_delta.sum(1)

        r_ec  = self.ec_head(ec_final.unsqueeze(1)).squeeze(1)
        r_hc  = self.hc_head(torch.stack([hc_min,  hc_final],  dim=1)).squeeze(1)
        r_exh = self.exh_head(torch.stack([exh_min, exh_final], dim=1)).squeeze(1)

        return torch.stack([r_ec, r_hc, r_exh], dim=1)  # (B, 3)


# ── training ──────────────────────────────────────────────────────────────────

def run_fold(lv12, train_idx, val_idx, args):
    train_dl = DataLoader(ChartDataset(lv12, train_idx, augment=True),
                          batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(ChartDataset(lv12, val_idx, augment=False),
                          batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0)

    model = GaugeSimHead(dropout=args.dropout).to(DEVICE)
    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_val, best_state = float('inf'), None
    for epoch in tqdm(range(1, args.epochs + 1), desc='  training', leave=False):
        model.train()
        for x, nc, nt, mask, y, _ in train_dl:
            x, nc, nt, mask, y = (x.to(DEVICE), nc.to(DEVICE), nt.to(DEVICE),
                                   mask.to(DEVICE), y.to(DEVICE))
            loss = nn.functional.mse_loss(model(x, nc, nt, mask), y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, nc, nt, mask, y, _ in val_dl:
                pred = model(x.to(DEVICE), nc.to(DEVICE), nt.to(DEVICE), mask.to(DEVICE))
                val_loss += nn.functional.mse_loss(pred, y.to(DEVICE)).item()
        val_loss /= len(val_dl)
        if val_loss < best_val:
            best_val   = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()
    preds, targets, chart_idxs = [], [], []
    with torch.no_grad():
        for x, nc, nt, mask, y, idxs in val_dl:
            preds.append(model(x.to(DEVICE), nc.to(DEVICE),
                               nt.to(DEVICE), mask.to(DEVICE)).cpu())
            targets.append(y)
            chart_idxs.extend(idxs)
    return torch.cat(preds).numpy(), torch.cat(targets).numpy(), chart_idxs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',     type=int,   default=100)
    p.add_argument('--batch-size', type=int,   default=32)
    p.add_argument('--lr',         type=float, default=1e-3)
    p.add_argument('--wd',         type=float, default=0.05)
    p.add_argument('--dropout',    type=float, default=0.2)
    p.add_argument('--folds',      type=int,   default=5)
    args = p.parse_args()

    print(f'device: {DEVICE}')
    manifest = pd.read_csv(MANIFEST)
    lv12     = build_cache(manifest)

    valid_idx = [i for i in range(len(lv12)) if (CACHE_DIR / f'{i}.npz').exists()]
    print(f'charts with cache: {len(valid_idx)}')

    n_params = sum(p.numel() for p in GaugeSimHead().parameters())
    print(f'GaugeSimHead params: {n_params:,}  (hardness MLP + 3 linear rating heads)')

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_preds, all_targets, all_idxs = [], [], []

    for fold, (tr, va) in enumerate(kf.split(valid_idx), 1):
        train_idx = [valid_idx[i] for i in tr]
        val_idx   = [valid_idx[i] for i in va]
        print(f'\nfold {fold}/{args.folds}  train={len(train_idx)}  val={len(val_idx)}')
        preds, targets, idxs = run_fold(lv12, train_idx, val_idx, args)
        all_preds.append(preds)
        all_targets.append(targets)
        all_idxs.extend(idxs)
        for j, name in enumerate(['ec', 'hc', 'exh']):
            mae = np.mean(np.abs(preds[:, j] - targets[:, j]))
            rho = spearmanr(preds[:, j], targets[:, j]).statistic
            print(f'  {name}: MAE={mae:.4f}  ρ={rho:.4f}')

    print('\n── 5-fold aggregate ──────────────────────────────')
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    for j, name in enumerate(['ec', 'hc', 'exh']):
        mae = np.mean(np.abs(all_preds[:, j] - all_targets[:, j]))
        rho = spearmanr(all_preds[:, j], all_targets[:, j]).statistic
        print(f'  {name}: MAE={mae:.4f}  ρ={rho:.4f}')

    # ── per-chart predictions ─────────────────────────────────────────────────
    rows = []
    for k, i in enumerate(all_idxs):
        r = lv12.iloc[i]
        rows.append({
            'title':    r['title'],
            'diftype':  r['diftype'],
            'ec':       all_targets[k, 0],
            'hc':       all_targets[k, 1],
            'exh':      all_targets[k, 2],
            'ec_pred':  all_preds[k, 0],
            'hc_pred':  all_preds[k, 1],
            'exh_pred': all_preds[k, 2],
        })
    df = pd.DataFrame(rows)
    df['ec_err']  = df['ec_pred']  - df['ec']
    df['hc_err']  = df['hc_pred']  - df['hc']
    df['exh_err'] = df['exh_pred'] - df['exh']
    df.to_csv('predictions_temporal.csv', index=False)
    print('\npredictions saved → predictions_temporal.csv')

    w = 46
    for gauge, col in [('EC', 'ec_err'), ('HC', 'hc_err'), ('EXH', 'exh_err')]:
        print(f'\n── {gauge} outliers ──────────────────────────────────────────────────────')
        print(f'  {"title":<{w}}  {"diftype":<16}  {"actual":>6}  {"pred":>6}  {"err":>6}')
        print(f'  {"─"*w}  {"─"*16}  {"─"*6}  {"─"*6}  {"─"*6}')
        top = df.reindex(df[col].abs().nlargest(20).index)
        gi  = ['ec', 'hc', 'exh'][['EC', 'HC', 'EXH'].index(gauge)]
        for _, row in top.iterrows():
            print(f'  {str(row["title"]):<{w}}  {str(row["diftype"]):<16}'
                  f'  {row[gi]:>6.2f}  {row[gi+"_pred"]:>6.2f}  {row[col]:>+6.2f}')


if __name__ == '__main__':
    main()
