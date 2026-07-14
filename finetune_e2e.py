"""
End-to-end fine-tuning: v11 encoder (unfrozen, lr=1e-5) + TemporalHead (lr=1e-3).

Raw windowed chart data (P1+P2, float16) cached to cache/e2e_v11/.
5-fold CV on 684 rated lv12 charts.  Flip augmentation doubles training size.
Grad clip=1.0.  Warmup 5ep for encoder, then cosine.
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
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from dataset import (
    encode_chart_pretrain_v6, window_chart_time_half,
    PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
    NUM_PATCHES_V10, PATCH_ROWS_V7,
    WIN_SECS_V10, WIN_ROWS_V10,
)
from mae import MAEModel

MANIFEST  = Path('/home/jysuh/projects/iidx_data/labeled_manifest.csv')
DATA_ROOT = Path('/home/jysuh/projects/iidx_data')
CKPT      = Path('checkpoints/pretrain_half_v11/encoder_best.pt')
CACHE_DIR = Path('cache/e2e_v11')
DEVICE    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_DIM = 192


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
    return model.encoder  # NOT .eval() — we'll fine-tune it


# ── raw-window cache ──────────────────────────────────────────────────────────

def build_cache(manifest):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    missing = [i for i in range(len(lv12))
               if not (CACHE_DIR / f'{i}.npz').exists()]
    if not missing:
        print(f'cache complete ({len(lv12)} charts)')
        return lv12

    print(f'building cache: {len(missing)} charts...')
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
        # Store lane channels only (cols 0:8); BPM not used by v11 encoder
        np.savez(CACHE_DIR / f'{i}.npz',
                 p1=p1_wins[:, :, :8].astype(np.float16),
                 p2=p2_wins[:, :, :8].astype(np.float16),
                 win_nc=win_nc.astype(np.float32))
    return lv12


# ── dataset ───────────────────────────────────────────────────────────────────

class ChartDataset(Dataset):
    def __init__(self, lv12, indices, augment=True):
        self.samples = []
        for i in indices:
            p = CACHE_DIR / f'{i}.npz'
            if not p.exists():
                continue
            row  = lv12.iloc[i]
            data = np.load(p)
            p1   = data['p1'].astype(np.float32)   # (T, WIN_ROWS_V10, 8)
            p2   = data['p2'].astype(np.float32)
            ec   = float(row['rating_ec'])
            hc   = float(row['rating_hc'])  if pd.notna(row['rating_hc'])  else ec
            exh  = float(row['rating_exh']) if pd.notna(row['rating_exh']) else ec
            y    = np.array([ec, hc, exh], dtype=np.float32)
            self.samples.append((p1, p2, y, i))
            if augment:
                self.samples.append((p2, p1, y, i))   # flip: swap sides

    def __len__(self):        return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def collate(batch):
    p1s, p2s, ys, idxs = zip(*batch)
    lengths = [p.shape[0] for p in p1s]
    T_max   = max(lengths)
    B, R, C = len(p1s), p1s[0].shape[1], p1s[0].shape[2]

    padded_p1 = torch.zeros(B, T_max, R, C)
    padded_p2 = torch.zeros(B, T_max, R, C)
    mask      = torch.ones(B, T_max, dtype=torch.bool)   # True = padded

    for i, (p1, p2, l) in enumerate(zip(p1s, p2s, lengths)):
        padded_p1[i, :l] = torch.from_numpy(p1)
        padded_p2[i, :l] = torch.from_numpy(p2)
        mask[i, :l]      = False

    return (padded_p1, padded_p2, mask,
            torch.from_numpy(np.stack(ys)),
            list(idxs))


# ── model ─────────────────────────────────────────────────────────────────────

class TemporalHead(nn.Module):
    def __init__(self, in_dim=385, model_dim=64, n_heads=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.proj = nn.Linear(in_dim, model_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim, nhead=n_heads,
            dim_feedforward=model_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm  = nn.LayerNorm(model_dim)
        self.heads = nn.ModuleList([nn.Linear(model_dim, 1) for _ in range(3)])

    def forward(self, x, mask=None):
        h = self.proj(x)                                    # (B, T, model_dim)
        h = self.transformer(h, src_key_padding_mask=mask)
        h = self.norm(h)
        if mask is not None:
            h  = h.masked_fill(mask.unsqueeze(-1), 0.0)
            ln = (~mask).float().sum(1, keepdim=True).clamp(min=1)
            h  = h.sum(1) / ln
        else:
            h = h.mean(1)                                   # (B, model_dim)
        return torch.stack([hd(h).squeeze(-1) for hd in self.heads], dim=1)  # (B, 3)


class E2EModel(nn.Module):
    def __init__(self, encoder, head):
        super().__init__()
        self.encoder = encoder
        self.head    = head

    def forward(self, p1, p2, mask):
        # p1, p2: (B, T_max, WIN_ROWS_V10, 8)
        # mask:   (B, T_max)  True = padded
        B, T_max, R, C = p1.shape

        # Flatten batch×time, encode all windows in one GPU pass
        e1 = self.encoder.embed_segment(p1.view(B * T_max, R, C))  # (B*T, 192)
        e2 = self.encoder.embed_segment(p2.view(B * T_max, R, C))
        e1 = e1.view(B, T_max, EMBED_DIM)
        e2 = e2.view(B, T_max, EMBED_DIM)

        if mask is not None:
            m  = mask.unsqueeze(-1)
            e1 = e1.masked_fill(m, 0.0)
            e2 = e2.masked_fill(m, 0.0)

        embs = torch.cat([e1, e2], dim=-1)                  # (B, T_max, 384)
        pos  = torch.linspace(0, 1, T_max, device=p1.device)
        pos  = pos.unsqueeze(0).unsqueeze(-1).expand(B, -1, -1)  # (B, T_max, 1)
        x    = torch.cat([embs, pos], dim=-1)               # (B, T_max, 385)

        return self.head(x, mask)


# ── training ──────────────────────────────────────────────────────────────────

def run_fold(lv12, train_idx, val_idx, args):
    train_dl = DataLoader(ChartDataset(lv12, train_idx, augment=True),
                          batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate, num_workers=0)
    val_dl   = DataLoader(ChartDataset(lv12, val_idx, augment=False),
                          batch_size=args.batch_size, shuffle=False,
                          collate_fn=collate, num_workers=0)

    encoder = load_encoder().to(DEVICE)
    head    = TemporalHead(dropout=args.dropout).to(DEVICE)
    model   = E2EModel(encoder, head)

    # Separate LRs: encoder much lower to preserve pretrained representations
    opt = optim.AdamW([
        {'params': encoder.parameters(), 'lr': args.encoder_lr},
        {'params': head.parameters(),    'lr': args.head_lr},
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

    sched = optim.lr_scheduler.LambdaLR(opt, [enc_lr_lambda, head_lr_lambda])

    best_val, best_state = float('inf'), None
    scaler = torch.amp.GradScaler('cuda')
    use_amp = DEVICE.type == 'cuda'

    for epoch in tqdm(range(1, args.epochs + 1), desc='  training', leave=False):
        model.train()
        for p1, p2, mask, y, _ in train_dl:
            p1, p2, mask, y = (p1.to(DEVICE), p2.to(DEVICE),
                                mask.to(DEVICE), y.to(DEVICE))
            opt.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                loss = nn.functional.mse_loss(model(p1, p2, mask), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sched.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for p1, p2, mask, y, _ in val_dl:
                with torch.amp.autocast('cuda', enabled=use_amp):
                    pred = model(p1.to(DEVICE), p2.to(DEVICE), mask.to(DEVICE))
                val_loss += nn.functional.mse_loss(pred, y.to(DEVICE)).item()
        val_loss /= len(val_dl)
        if val_loss < best_val:
            best_val  = val_loss
            best_state = {
                'encoder': {k: v.clone() for k, v in encoder.state_dict().items()},
                'head':    {k: v.clone() for k, v in head.state_dict().items()},
            }

    encoder.load_state_dict(best_state['encoder'])
    head.load_state_dict(best_state['head'])
    model.eval()

    preds, targets, chart_idxs = [], [], []
    with torch.no_grad():
        for p1, p2, mask, y, idxs in val_dl:
            preds.append(model(p1.to(DEVICE), p2.to(DEVICE), mask.to(DEVICE)).cpu())
            targets.append(y)
            chart_idxs.extend(idxs)

    return torch.cat(preds).numpy(), torch.cat(targets).numpy(), chart_idxs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--epochs',        type=int,   default=100)
    p.add_argument('--batch-size',    type=int,   default=8)
    p.add_argument('--encoder-lr',    type=float, default=1e-5)
    p.add_argument('--head-lr',       type=float, default=1e-3)
    p.add_argument('--wd',            type=float, default=0.05)
    p.add_argument('--dropout',       type=float, default=0.2)
    p.add_argument('--warmup-epochs', type=int,   default=5)
    p.add_argument('--folds',         type=int,   default=5)
    p.add_argument('--start-fold',    type=int,   default=1,
                   help='resume from this fold (1-based); loads saved results for earlier folds')
    args = p.parse_args()

    print(f'device: {DEVICE}')
    manifest = pd.read_csv(MANIFEST)
    lv12     = build_cache(manifest)

    valid_idx = [i for i in range(len(lv12)) if (CACHE_DIR / f'{i}.npz').exists()]
    print(f'charts with cache: {len(valid_idx)}')

    enc_params  = sum(p.numel() for p in load_encoder().parameters())
    head_params = sum(p.numel() for p in TemporalHead().parameters())
    print(f'encoder params: {enc_params:,}  head params: {head_params:,}', flush=True)

    kf = KFold(n_splits=args.folds, shuffle=True, random_state=42)
    all_preds, all_targets, all_idxs = [], [], []

    for fold, (tr, va) in enumerate(kf.split(valid_idx), 1):
        fold_cache = Path(f'predictions_e2e_fold{fold}.npz')

        if fold < args.start_fold:
            if fold_cache.exists():
                saved   = np.load(fold_cache, allow_pickle=True)
                preds   = saved['preds']
                targets = saved['targets']
                idxs    = list(saved['idxs'])
                print(f'\nfold {fold}/{args.folds}  (loaded from {fold_cache})', flush=True)
            else:
                print(f'\nfold {fold}/{args.folds}  (skipped — no saved results)', flush=True)
                continue
        else:
            train_idx = [valid_idx[i] for i in tr]
            val_idx   = [valid_idx[i] for i in va]
            print(f'\nfold {fold}/{args.folds}  train={len(train_idx)}  val={len(val_idx)}', flush=True)
            preds, targets, idxs = run_fold(lv12, train_idx, val_idx, args)
            np.savez(fold_cache, preds=preds, targets=targets, idxs=np.array(idxs))

        all_preds.append(preds)
        all_targets.append(targets)
        all_idxs.extend(idxs)
        for j, name in enumerate(['ec', 'hc', 'exh']):
            mae = np.mean(np.abs(preds[:, j] - targets[:, j]))
            rho = spearmanr(preds[:, j], targets[:, j]).statistic
            print(f'  {name}: MAE={mae:.4f}  ρ={rho:.4f}', flush=True)

    n_agg = len(all_preds)
    print(f'\n── {n_agg}-fold aggregate ──────────────────────────────')
    all_preds   = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    for j, name in enumerate(['ec', 'hc', 'exh']):
        mae = np.mean(np.abs(all_preds[:, j] - all_targets[:, j]))
        rho = spearmanr(all_preds[:, j], all_targets[:, j]).statistic
        print(f'  {name}: MAE={mae:.4f}  ρ={rho:.4f}')

    rows = []
    for k, i in enumerate(all_idxs):
        r = lv12.iloc[i]
        rows.append({
            'title':    r['title'],   'diftype':  r['diftype'],
            'ec':       all_targets[k, 0], 'hc': all_targets[k, 1],
            'exh':      all_targets[k, 2],
            'ec_pred':  all_preds[k, 0],   'hc_pred': all_preds[k, 1],
            'exh_pred': all_preds[k, 2],
        })
    df = pd.DataFrame(rows)
    df['ec_err']  = df['ec_pred']  - df['ec']
    df['hc_err']  = df['hc_pred']  - df['hc']
    df['exh_err'] = df['exh_pred'] - df['exh']
    df.to_csv('predictions_e2e.csv', index=False)
    print('\npredictions saved → predictions_e2e.csv')

    print('\n── EC outliers (|err| > 4) ──────────────────────')
    out = df[df['ec_err'].abs() > 4].sort_values('ec_err')
    for _, r in out.iterrows():
        print(f'  {r["title"]:<45} {r["diftype"]:<20}'
              f' ec={r["ec"]:.1f}  pred={r["ec_pred"]:.1f}  err={r["ec_err"]:+.1f}')


if __name__ == '__main__':
    main()
