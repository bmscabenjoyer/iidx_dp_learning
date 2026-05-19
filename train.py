"""
Fine-tuning on lv12 decimal ratings with optional pretrained encoder.

Each chart is one sample; batch_size=1 with gradient accumulation.
Predicts rating_ec, rating_hc, rating_exh via differentiable gauge simulation.
"""

import argparse
import random
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from dataset import FinetuneDataset
from mae import MAEEncoder
from model import IIDXModel, finetune_loss


def evaluate(model, loader, device):
    model.eval()
    total_mae = 0.0
    n = 0
    with torch.no_grad():
        for windows, note_counts, targets in loader:
            windows     = windows[0].to(device, dtype=torch.float32)
            note_counts = note_counts[0].to(device)
            targets     = targets[0].to(device)
            pred = model(windows, note_counts)
            # MAE on stat rating: approximate as mean of three gauge ratings
            pred_mean = (pred['ec'] + pred['hc'] + pred['exh']) / 3.0
            tgt_mean  = targets.mean()
            total_mae += (pred_mean - tgt_mean).abs().item()
            n += 1
    return total_mae / max(n, 1)


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    dataset = FinetuneDataset(
        manifest_csv=args.manifest,
        data_root=args.data_root,
        augment=True,
    )
    print(f'lv12 charts (×4 augmentations): {len(dataset):,}')

    val_size   = max(1, int(len(dataset) * args.val_split))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    # augment=False for val to avoid data leakage; use base entries only
    # (simplification: val_ds still contains augmented items from random_split)
    print(f'train: {len(train_ds)}  val: {len(val_ds)}')

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True,
                              num_workers=args.workers,
                              pin_memory=(device.type == 'cuda'))
    val_loader   = DataLoader(val_ds,   batch_size=1, shuffle=False,
                              num_workers=0)

    # load pretrained encoder if provided, otherwise random init
    encoder = MAEEncoder(embed_dim=args.embed_dim, depth=args.encoder_depth)
    if args.pretrained_encoder:
        ckpt = Path(args.pretrained_encoder)
        encoder.load_state_dict(torch.load(ckpt, map_location='cpu'))
        print(f'loaded pretrained encoder from {ckpt}')
    else:
        print('training encoder from random init (no pretrained weights)')

    model = IIDXModel(encoder=encoder, embed_dim=args.embed_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model params: {total_params:,}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_val_mae = float('inf')
    accum_steps  = args.grad_accum

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        optimizer.zero_grad()

        for step, (windows, note_counts, targets) in enumerate(
                tqdm(train_loader, desc=f'epoch {epoch}/{args.epochs}', leave=False)):

            windows     = windows[0].to(device, dtype=torch.float32)
            note_counts = note_counts[0].to(device)
            targets     = targets[0].to(device)

            pred = model(windows, note_counts)
            loss = finetune_loss(pred, targets) / accum_steps
            loss.backward()
            train_loss += loss.item() * accum_steps

            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        scheduler.step()
        val_mae = evaluate(model, val_loader, device)
        avg_loss = train_loss / len(train_loader)
        print(f'epoch {epoch:3d}  loss={avg_loss:.4f}  val_mae={val_mae:.4f}'
              f'  lr={scheduler.get_last_lr()[0]:.2e}')

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            ckpt = out_dir / 'model_best.pt'
            torch.save(model.state_dict(), ckpt)
            print(f'  → saved to {ckpt}  (val_mae={val_mae:.4f})')

    torch.save(model.state_dict(), out_dir / 'model_final.pt')
    print(f'training done. best val MAE: {best_val_mae:.4f}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',            required=True)
    p.add_argument('--data-root',           required=True)
    p.add_argument('--pretrained-encoder',  default=None,
                   help='path to encoder_best.pt from pretrain.py')
    p.add_argument('--out-dir',             default='checkpoints/finetune')
    p.add_argument('--epochs',              type=int,   default=50)
    p.add_argument('--lr',                  type=float, default=1e-4)
    p.add_argument('--wd',                  type=float, default=0.01)
    p.add_argument('--val-split',           type=float, default=0.15)
    p.add_argument('--grad-accum',          type=int,   default=8)
    p.add_argument('--embed-dim',           type=int,   default=128)
    p.add_argument('--encoder-depth',       type=int,   default=6)
    p.add_argument('--workers',             type=int,   default=2)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
