"""
LoRA fine-tuning of the half-chart encoder for EC/HC/EXH rating prediction.

LoRA is applied to each transformer layer's FFN (linear1, linear2) and
attention out_proj — ~147K trainable LoRA params + ~34K head = ~181K total
out of 5M frozen encoder params (3.6%).

k=5 position-weighted mean pooling, [mean, diff] P1+P2 fusion → 520-dim → head.
5-fold CV vs frozen Ridge k=5 baseline (EC ρ=0.772, HC ρ=0.775, EXH ρ=0.738).
"""

import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
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
NOTE_DIM   = 8
K_WEIGHT   = 5

LORA_RANK  = 8
LORA_ALPHA = 16
ENC_LR     = 5e-5
HEAD_LR    = 5e-4
WD         = 1e-2
EPOCHS     = 150
PATIENCE   = 20

BASELINE_RHO = {'rating_ec': 0.7722, 'rating_hc': 0.7747, 'rating_exh': 0.7376}
BASELINE_MAE = {'rating_ec': 1.3888, 'rating_hc': 1.0200, 'rating_exh': 0.7780}
XGB_RHO      = {'rating_ec': 0.7303, 'rating_hc': 0.7316, 'rating_exh': 0.6920}


# ── LoRA ──────────────────────────────────────────────────────────────────────

class LoRALinear(nn.Module):
    """Frozen Linear + trainable low-rank update: h = (W + scale*BA)x."""
    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        for p in self.linear.parameters():
            p.requires_grad_(False)
        self.A     = nn.Parameter(torch.randn(rank, linear.in_features) * 0.02)
        self.B     = nn.Parameter(torch.zeros(linear.out_features, rank))
        self.scale = alpha / rank

    @property
    def weight(self) -> torch.Tensor:
        # PyTorch MHA accesses out_proj.weight directly; expose effective weight.
        return self.linear.weight + self.scale * (self.B @ self.A)

    @property
    def bias(self):
        return self.linear.bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


def inject_lora(encoder: nn.Module, rank: int = LORA_RANK,
                alpha: float = LORA_ALPHA) -> nn.Module:
    """
    Freeze all encoder params, inject LoRA into each transformer layer:
      - FFN linear1 and linear2
      - Attention out_proj
    Tiny projection layers (density_proj, bpm_proj) are fully unfrozen.
    """
    for p in encoder.parameters():
        p.requires_grad_(False)

    for layer in encoder.transformer.layers:
        layer.linear1                = LoRALinear(layer.linear1,                rank, alpha)
        layer.linear2                = LoRALinear(layer.linear2,                rank, alpha)
        layer.self_attn.out_proj     = LoRALinear(layer.self_attn.out_proj,     rank, alpha)

    # fully unfreeze tiny conditioning projections (256 params each)
    for p in encoder.density_proj.parameters():
        p.requires_grad_(True)
    if hasattr(encoder, 'bpm_proj'):
        for p in encoder.bpm_proj.parameters():
            p.requires_grad_(True)

    return encoder


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── Rating model ──────────────────────────────────────────────────────────────

class RatingHead(nn.Module):
    def __init__(self, in_dim: int = EMBED_DIM * 2 + NOTE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Data loading ──────────────────────────────────────────────────────────────

def precompute(manifest: pd.DataFrame):
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        manifest['rating_hc'].notna() &
        manifest['rating_exh'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    data, skipped = [], 0
    for _, row in tqdm(lv12.iterrows(), total=len(lv12), desc='loading charts'):
        npy = DATA_ROOT / 'dp12_active' / str(row['file_path'])
        if not npy.exists():
            skipped += 1; continue
        enc, nc, _ = encode_chart_pretrain_v6(str(npy))
        p1_wins, p2_wins, win_nc = window_chart_split(enc, nc, WINDOW_BARS, STRIDE_BARS)
        if p1_wins.shape[0] == 0:
            skipped += 1; continue

        T = p1_wins.shape[0]
        t = np.arange(T, dtype=np.float32)
        w = 1.0 + K_WEIGHT * (t / max(T - 1, 1))
        w = torch.from_numpy(w / w.sum()).float()  # (T,) normalised weights

        nc_arr = win_nc.astype(np.float32)
        last_q = nc_arr[max(0, int(len(nc_arr) * 0.75)):]
        note_feats = torch.tensor([
            nc_arr.mean(), nc_arr.std() + 1e-6, nc_arr.max(),
            np.percentile(nc_arr, 75), np.percentile(nc_arr, 25),
            last_q.mean() / (nc_arr.mean() + 1e-6),
            (nc_arr.std() + 1e-6) / (nc_arr.mean() + 1e-6),
            nc_arr.sum(),
        ], dtype=torch.float32)

        data.append({
            'p1_x':   torch.from_numpy(p1_wins[:, :, :8]).float(),  # (T, 768, 8)
            'p2_x':   torch.from_numpy(p2_wins[:, :, :8]).float(),
            'p1_bpm': torch.from_numpy(p1_wins[:, :, 8].mean(1)).float(),  # (T,)
            'p2_bpm': torch.from_numpy(p2_wins[:, :, 8].mean(1)).float(),
            'w':      w,           # (T,) position weights
            'note':   note_feats,  # (8,)
            'labels': torch.tensor(
                [float(row['rating_ec']), float(row['rating_hc']), float(row['rating_exh'])],
                dtype=torch.float32,
            ),
        })

    print(f'loaded {len(data)} charts  (skipped {skipped})', flush=True)
    return data


def embed_chart(encoder, item):
    """Forward both hands through encoder, apply weighted mean, fuse [mean, diff]."""
    p1_cls = encoder.embed_segment(
        item['p1_x'].to(DEVICE), bpm=item['p1_bpm'].to(DEVICE)
    )  # (T, 256)
    p2_cls = encoder.embed_segment(
        item['p2_x'].to(DEVICE), bpm=item['p2_bpm'].to(DEVICE)
    )
    w = item['w'].to(DEVICE)                   # (T,)
    p1_m = (p1_cls * w[:, None]).sum(0)        # (256,)
    p2_m = (p2_cls * w[:, None]).sum(0)
    return torch.cat([(p1_m + p2_m) / 2.0, p1_m - p2_m])  # (512,)


# ── Training ──────────────────────────────────────────────────────────────────

def train_fold(train_data, val_data, base_encoder):
    encoder = copy.deepcopy(base_encoder)
    inject_lora(encoder)
    encoder = encoder.to(DEVICE)
    head    = RatingHead().to(DEVICE)

    lora_params = [p for p in encoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([
        {'params': lora_params,            'lr': ENC_LR},
        {'params': head.parameters(),      'lr': HEAD_LR},
    ], weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    train_labels = torch.stack([d['labels'] for d in train_data]).numpy()
    y_std = torch.tensor(train_labels.std(0), dtype=torch.float32,
                         device=DEVICE).clamp(min=0.1)

    best_val, best_state, patience_ctr = float('inf'), None, 0
    rng = np.random.default_rng(42)

    for epoch in range(1, EPOCHS + 1):
        encoder.train(); head.train()
        total_loss = 0.0

        for i in rng.permutation(len(train_data)):
            item = train_data[i]
            emb  = embed_chart(encoder, item)                    # (512,)
            nf   = item['note'].to(DEVICE)
            pred = head(torch.cat([emb, nf]).unsqueeze(0))       # (1, 3)
            tgt  = item['labels'].to(DEVICE).unsqueeze(0)
            loss = ((pred - tgt) / y_std).pow(2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(
                list(lora_params) + list(head.parameters()), 1.0)
            opt.step()
            total_loss += loss.item()
        sched.step()

        encoder.eval(); head.eval()
        val_loss = 0.0
        with torch.no_grad():
            for item in val_data:
                emb  = embed_chart(encoder, item)
                nf   = item['note'].to(DEVICE)
                pred = head(torch.cat([emb, nf]).unsqueeze(0))
                tgt  = item['labels'].to(DEVICE).unsqueeze(0)
                val_loss += ((pred - tgt) / y_std).pow(2).mean().item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone()
                          for k, v in list(encoder.state_dict().items()) +
                                       list({'head.' + k: v
                                             for k, v in head.state_dict().items()}.items())}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if epoch % 10 == 0:
            print(f'    ep {epoch:3d}  train={total_loss/len(train_data):.4f}'
                  f'  val={val_loss/len(val_data):.4f}'
                  f'  best={best_val/len(val_data):.4f}', flush=True)

        if patience_ctr >= PATIENCE:
            print(f'    early stop at ep {epoch}', flush=True)
            break

    # restore best
    enc_state  = {k: v for k, v in best_state.items() if not k.startswith('head.')}
    head_state = {k[5:]: v for k, v in best_state.items() if k.startswith('head.')}
    encoder.load_state_dict({k: v.to(DEVICE) for k, v in enc_state.items()})
    head.load_state_dict({k: v.to(DEVICE) for k, v in head_state.items()})
    return encoder, head


def evaluate(encoder, head, data):
    encoder.eval(); head.eval()
    preds, trues = {t: [] for t in TARGETS}, {t: [] for t in TARGETS}
    with torch.no_grad():
        for item in data:
            emb  = embed_chart(encoder, item)
            nf   = item['note'].to(DEVICE)
            pred = head(torch.cat([emb, nf]).unsqueeze(0)).cpu().numpy()[0]
            for i, t in enumerate(TARGETS):
                preds[t].append(float(pred[i]))
                trues[t].append(float(item['labels'][i]))
    return preds, trues


# ── Main ──────────────────────────────────────────────────────────────────────

def run(manifest_path=None, data_root=None, ckpt_path=None):
    global MANIFEST, DATA_ROOT, CKPT
    if manifest_path: MANIFEST  = Path(manifest_path)
    if data_root:     DATA_ROOT = Path(data_root)
    if ckpt_path:     CKPT      = Path(ckpt_path)

    print(f'device: {DEVICE}', flush=True)

    base_model = MAEModel(
        in_channels=PRETRAIN_IN_CHANNELS_HALF,
        num_patches=NUM_PATCHES_V4, patch_rows=PATCH_ROWS_V4,
        encoder_dim=EMBED_DIM, decoder_dim=128, num_heads=8,
        use_lane_type_embed=True, use_bpm_cond=True,
        lane_types=LANE_TYPES_HALF,
    )
    base_model.encoder.load_state_dict(torch.load(CKPT, map_location='cpu'))
    base_encoder = base_model.encoder

    _enc_example = copy.deepcopy(base_encoder)
    inject_lora(_enc_example)
    _head_example = RatingHead()
    n_lora = count_trainable(_enc_example)
    n_head = count_trainable(_head_example)
    print(f'trainable: {n_lora:,} (LoRA) + {n_head:,} (head) = {n_lora+n_head:,} '
          f'/ {sum(p.numel() for p in base_encoder.parameters()):,} total encoder params',
          flush=True)
    del _enc_example, _head_example

    manifest = pd.read_csv(MANIFEST)
    data     = precompute(manifest)

    kf        = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_maes = {t: [] for t in TARGETS}
    fold_rhos = {t: [] for t in TARGETS}

    for fold, (tr_idx, va_idx) in enumerate(kf.split(data), 1):
        train_data = [data[i] for i in tr_idx]
        val_data   = [data[i] for i in va_idx]
        print(f'\n── fold {fold} ──', flush=True)

        encoder, head = train_fold(train_data, val_data, base_encoder)
        preds, trues  = evaluate(encoder, head, val_data)

        parts = [f'fold {fold}:']
        for t in TARGETS:
            p   = np.array(preds[t]); y = np.array(trues[t])
            mae = float(np.abs(p - y).mean())
            rho = float(spearmanr(y, p).statistic)
            fold_maes[t].append(mae); fold_rhos[t].append(rho)
            parts.append(f'  {t.split("_")[1].upper()} MAE={mae:.4f} ρ={rho:.3f}')
        print(''.join(parts), flush=True)

    print('\n── Results ─────────────────────────────────────────────────────────────', flush=True)
    print(f'{"target":<12}  {"XGB ρ":>7}  {"frozen k=5":>10}  {"LoRA ρ":>8}  '
          f'{"frozen MAE":>11}  {"LoRA MAE":>10}')
    print('─' * 68)
    for t in TARGETS:
        rho = np.mean(fold_rhos[t]); mae = np.mean(fold_maes[t])
        print(f'{t:<12}  {XGB_RHO[t]:7.4f}  {BASELINE_RHO[t]:10.4f}  {rho:8.4f}  '
              f'{BASELINE_MAE[t]:11.4f}  {mae:10.4f}', flush=True)


if __name__ == '__main__':
    run()
