"""MAE pretraining on all chart segments (lv10/11/12, no labels required)."""

import argparse
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import PretrainDataset
from mae import MAEModel


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    dataset = PretrainDataset(
        manifest_csv=args.manifest,
        data_root=args.data_root,
        augment=True,
    )
    print(f'pretraining windows: {len(dataset):,}')

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=2 if args.workers > 0 else None,
        persistent_workers=(args.workers > 0),
    )

    model = MAEModel(
        mask_ratio=args.mask_ratio,
        encoder_dim=args.embed_dim,
        decoder_dim=args.decoder_dim,
        encoder_depth=args.encoder_depth,
        decoder_depth=args.decoder_depth,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'model params: {total_params:,}')

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(loader, desc=f'epoch {epoch}/{args.epochs}', leave=False):
            x = batch.to(device, dtype=torch.float32)
            loss = model(x)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        scheduler.step()
        print(f'epoch {epoch:3d}  loss={avg_loss:.5f}  lr={scheduler.get_last_lr()[0]:.2e}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            ckpt = out_dir / 'encoder_best.pt'
            torch.save(model.encoder.state_dict(), ckpt)
            print(f'  → saved encoder to {ckpt}')

    # always save final
    torch.save(model.encoder.state_dict(), out_dir / 'encoder_final.pt')
    print(f'pretraining done. best loss: {best_loss:.5f}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',       required=True,  help='path to labeled_manifest.csv')
    p.add_argument('--data-root',      required=True,  help='path to iidx_data/')
    p.add_argument('--out-dir',        default='checkpoints/pretrain')
    p.add_argument('--epochs',         type=int,   default=100)
    p.add_argument('--batch-size',     type=int,   default=256)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--mask-ratio',     type=float, default=0.75)
    p.add_argument('--embed-dim',      type=int,   default=128)
    p.add_argument('--decoder-dim',    type=int,   default=64)
    p.add_argument('--encoder-depth',  type=int,   default=6)
    p.add_argument('--decoder-depth',  type=int,   default=2)
    p.add_argument('--workers',        type=int,   default=4)
    args = p.parse_args()
    train(args)


if __name__ == '__main__':
    main()
