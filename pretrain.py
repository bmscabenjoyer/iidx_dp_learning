"""MAE pretraining on all chart segments (lv10/11/12, no labels required)."""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from dataset import (PretrainDataset, PRETRAIN_IN_CHANNELS_V3,
                     PATCH_ROWS, NUM_PATCHES, PATCH_ROWS_V4, NUM_PATCHES_V4)
from mae import MAEModel


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    dataset = PretrainDataset(
        manifest_csv=args.manifest,
        data_root=args.data_root,
        augment=True,
        zasa_csv=args.zasa_csv,
        encoding_v3=args.v3 or args.v4,
    )
    print(f'pretraining windows: {len(dataset):,}')

    # Density-weighted sampler.
    # __getitem__ maps idx → base_idx = idx // 8, so aug variants of the same
    # base window are at consecutive indices.  Interleave weights accordingly.
    base_weights = dataset.window_weights          # list[float], one per base window
    aug_weights  = [w for w in base_weights for _ in range(8)]
    sampler = WeightedRandomSampler(
        weights=aug_weights,
        num_samples=len(dataset),
        replacement=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=4 if args.workers > 0 else None,
        persistent_workers=(args.workers > 0),
    )

    in_ch       = PRETRAIN_IN_CHANNELS_V3 if (args.v3 or args.v4) else 32
    patch_rows  = PATCH_ROWS_V4  if args.v4 else PATCH_ROWS
    num_patches = NUM_PATCHES_V4 if args.v4 else NUM_PATCHES

    model = MAEModel(
        mask_ratio=args.mask_ratio,
        encoder_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
        in_channels=in_ch,
        num_patches=num_patches,
        patch_rows=patch_rows,
        nonzero_weight=args.nonzero_weight,
        hold_weight=args.hold_weight,
        event_weight=args.event_weight if args.v4 else 1.0,
        empty_weight=args.empty_weight if args.v4 else 1.0,
        density_mask=args.v4,
        zasa_weight=args.zasa_weight,
    ).to(device)

    if args.compile:
        model = torch.compile(model)
        print('model compiled')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model params: {total_params:,}  nonzero_weight={args.nonzero_weight}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.amp.GradScaler(device='cuda', enabled=(device.type == 'cuda'))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss  = 0.0
        total_recon = 0.0
        total_zasa  = 0.0
        for batch in tqdm(loader, desc=f'epoch {epoch}/{args.epochs}', leave=False):
            x, density, zasa_label = batch
            x          = x.to(device, dtype=torch.float32)
            density    = density.to(device, dtype=torch.float32)
            zasa_label = zasa_label.to(device, dtype=torch.float32)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                loss, recon_val, zasa_val = model(x, density, zasa_label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss  += loss.item()
            total_recon += recon_val
            total_zasa  += zasa_val

        avg_loss  = total_loss  / len(loader)
        avg_recon = total_recon / len(loader)
        avg_zasa  = total_zasa  / len(loader)
        scheduler.step()
        print(f'epoch {epoch:3d}  loss={avg_loss:.5f}  recon={avg_recon:.5f}  '
              f'zasa={avg_zasa:.5f}  lr={scheduler.get_last_lr()[0]:.2e}', flush=True)

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = out_dir / 'encoder_best.pt'
            torch.save(model.encoder.state_dict(), ckpt)
            print(f'  → saved encoder to {ckpt}')

    torch.save(model.encoder.state_dict(), out_dir / 'encoder_final.pt')
    print(f'pretraining done. best loss: {best_loss:.5f}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',        required=True,  help='path to labeled_manifest.csv')
    p.add_argument('--data-root',       required=True,  help='path to iidx_data/')
    p.add_argument('--out-dir',         default='checkpoints/pretrain')
    p.add_argument('--epochs',          type=int,   default=100)
    p.add_argument('--batch-size',      type=int,   default=256)
    p.add_argument('--lr',              type=float, default=1e-3)
    p.add_argument('--mask-ratio',      type=float, default=0.75)
    p.add_argument('--embed-dim',       type=int,   default=128)
    p.add_argument('--decoder-dim',     type=int,   default=64)
    p.add_argument('--encoder-depth',   type=int,   default=6)
    p.add_argument('--decoder-depth',   type=int,   default=2)
    p.add_argument('--workers',         type=int,   default=4)
    p.add_argument('--nonzero-weight',  type=float, default=4.0,
                   help='extra loss weight on tap cells (value ~1.0)')
    p.add_argument('--hold-weight',     type=float, default=2.0,
                   help='extra loss weight on hold body cells (value ~0.5, v3 only)')
    p.add_argument('--v3',              action='store_true',
                   help='use v3 single-channel encoding (16 lane cols, merged hold)')
    p.add_argument('--v4',              action='store_true',
                   help='v4: sub-beat patches (12 rows, 64 patches) + density masking + event loss')
    p.add_argument('--event-weight',    type=float, default=20.0,
                   help='loss weight for event rows (v4 only)')
    p.add_argument('--empty-weight',    type=float, default=1.0,
                   help='loss weight for empty rows (v4 only)')
    p.add_argument('--zasa-csv',        default=None,
                   help='path to zasa_ratings.csv for auxiliary difficulty supervision')
    p.add_argument('--zasa-weight',     type=float, default=0.1,
                   help='loss weight for auxiliary zasa regression head')
    p.add_argument('--compile',         action='store_true', help='torch.compile the model')
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
