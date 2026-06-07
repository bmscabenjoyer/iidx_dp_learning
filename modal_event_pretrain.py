"""
Modal training for IIDX DP event-token MAE pretraining.

Architecture
------------
Per-timestep event tokens (joint 16-lane) → EventEncoder (4-layer transformer,
128-dim CLS) → trained via masked autoencoding on lane_bits / delta_bin /
note_type reconstruction.

Setup (one-time):
    modal volume create iidx-data    # already exists
    modal volume create iidx-ckpts   # already exists

Run:
    modal run modal_event_pretrain.py

Download checkpoint:
    modal volume get iidx-ckpts event_pretrain/encoder_best.pt \\
        checkpoints/event_pretrain/encoder_best.pt
"""

import modal

# ── Image ─────────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "tqdm")
    .pip_install("torch==2.5.1", index_url="https://download.pytorch.org/whl/cu124")
    .add_local_file("event_tokenize.py", "/app/event_tokenize.py")
    .add_local_file("event_dataset.py",  "/app/event_dataset.py")
    .add_local_file("event_encoder.py",  "/app/event_encoder.py")
)

# ── Volumes ───────────────────────────────────────────────────────────────────

data_vol = modal.Volume.from_name("iidx-data")
ckpt_vol = modal.Volume.from_name("iidx-ckpts", create_if_missing=True)

DATA_DIR = "/iidx-data"
CKPT_DIR = "/iidx-ckpts"

# ── App ───────────────────────────────────────────────────────────────────────

app = modal.App("iidx-event-pretrain")


@app.function(
    image=image,
    gpu="A10G",
    volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol},
    timeout=86400,
)
def train(
    levels:     list[int] = [10, 11, 12],
    d_model:    int   = 128,
    d_lane:     int   = 64,
    d_delta:    int   = 32,
    d_ntype:    int   = 32,
    n_layers:   int   = 4,
    n_heads:    int   = 4,
    mlp_ratio:  float = 4.0,
    dropout:    float = 0.1,
    win_sec:    float = 4.0,
    stride_sec: float = 2.0,
    max_tokens: int   = 128,
    mask_rate:  float = 0.30,
    augment:    bool  = True,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    weight_decay: float = 0.01,
    epochs:     int   = 200,
    warmup_epochs: int = 10,
    lane_w:          float = 1.00,
    delta_w:         float = 0.25,
    ntype_w:         float = 0.50,
    lane_pos_weight: float = 5.0,
    ckpt_name:       str   = "event_pretrain",
):
    import sys
    sys.path.insert(0, "/app")

    import math
    import time
    import torch
    from torch.utils.data import DataLoader, ConcatDataset
    from pathlib import Path

    from event_tokenize import build_cache
    from event_dataset  import EventMAEDataset
    from event_encoder  import EventMAEModel, model_info

    device = torch.device("cuda")

    # ── Build event token caches ───────────────────────────────────────────────

    cache_dirs = []
    for level in levels:
        chart_dir = f"{DATA_DIR}/dp{level}_active/charts"
        cache_dir = f"{DATA_DIR}/event_cache/dp{level}"
        print(f"Tokenising dp{level} charts → {cache_dir}")
        build_cache(chart_dir, cache_dir, overwrite=False)
        cache_dirs.append(cache_dir)

    # ── Dataset ────────────────────────────────────────────────────────────────

    datasets = []
    for cache_dir in cache_dirs:
        datasets.append(EventMAEDataset(
            cache_dir   = cache_dir,
            win_sec     = win_sec,
            stride_sec  = stride_sec,
            max_tokens  = max_tokens,
            mask_rate   = mask_rate,
            augment     = augment,
        ))

    dataset = ConcatDataset(datasets)
    print(f"Total windows: {len(dataset)}")

    loader = DataLoader(
        dataset,
        batch_size         = batch_size,
        shuffle            = True,
        num_workers        = 4,
        pin_memory         = True,
        drop_last          = True,
        persistent_workers = True,
    )

    # ── Model ──────────────────────────────────────────────────────────────────

    model = EventMAEModel(
        d_model          = d_model,
        d_lane           = d_lane,
        d_delta          = d_delta,
        d_ntype          = d_ntype,
        n_layers         = n_layers,
        n_heads          = n_heads,
        mlp_ratio        = mlp_ratio,
        dropout          = dropout,
        lane_w           = lane_w,
        delta_w          = delta_w,
        ntype_w          = ntype_w,
        lane_pos_weight  = lane_pos_weight,
    ).to(device)
    print(f"Model: {model_info(model)}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay)

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = Path(CKPT_DIR) / ckpt_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume from checkpoint if available ───────────────────────────────────

    start_epoch = 1
    best_loss   = float("inf")
    resume_path = ckpt_dir / "checkpoint_latest.pt"

    if resume_path.exists():
        print(f"Resuming from {resume_path} ...", flush=True)
        ckpt = torch.load(resume_path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_loss   = ckpt['best_loss']
        print(f"  resumed: epoch {ckpt['epoch']}, best_loss={best_loss:.4f}", flush=True)
    else:
        print("No checkpoint found — starting from scratch.", flush=True)

    # ── Training loop ──────────────────────────────────────────────────────────

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        totals = {
            'loss': 0.0, 'lane_loss': 0.0, 'delta_loss': 0.0, 'ntype_loss': 0.0,
            'lane_recall': 0.0, 'delta_acc': 0.0, 'ntype_acc': 0.0,
        }
        n_batches = 0
        t0 = time.time()

        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            loss, metrics = model(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in totals:
                totals[k] += metrics[k]
            n_batches += 1

        scheduler.step()
        epoch_secs = time.time() - t0

        avgs = {k: v / n_batches for k, v in totals.items()}
        avg_loss = avgs['loss']

        if epoch % 10 == 0 or epoch == 1:
            lr_now = optimizer.param_groups[0]['lr']
            print(
                f"epoch {epoch:>4}/{epochs}  "
                f"lr={lr_now:.2e}  "
                f"loss={avgs['loss']:.4f}  "
                f"lane={avgs['lane_loss']:.4f}(recall={avgs['lane_recall']:.3f})  "
                f"delta={avgs['delta_loss']:.4f}({avgs['delta_acc']:.3f})  "
                f"ntype={avgs['ntype_loss']:.4f}({avgs['ntype_acc']:.3f})  "
                f"t={epoch_secs:.1f}s",
                flush=True,
            )

        # Full training state — commit every 5 epochs for preemption recovery
        torch.save(
            {
                'epoch':           epoch,
                'model_state':     model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'best_loss':       best_loss,
            },
            ckpt_dir / "checkpoint_latest.pt",
        )

        new_best = avg_loss < best_loss
        if new_best:
            best_loss = avg_loss
            torch.save(
                {
                    'epoch':       epoch,
                    'state_dict':  model.encoder.state_dict(),
                    'loss':        best_loss,
                    'model_cfg': {
                        'd_model':   d_model,  'd_lane':  d_lane,
                        'd_delta':   d_delta,  'd_ntype': d_ntype,
                        'n_layers':  n_layers, 'n_heads': n_heads,
                        'mlp_ratio': mlp_ratio,
                    },
                },
                ckpt_dir / "encoder_best.pt",
            )

        if new_best or epoch % 5 == 0:
            ckpt_vol.commit()

    torch.save(
        {
            'epoch':      epochs,
            'state_dict': model.encoder.state_dict(),
            'loss':       avg_loss,
            'model_cfg': {
                'd_model':   d_model, 'd_lane':  d_lane,
                'd_delta':   d_delta, 'd_ntype': d_ntype,
                'n_layers':  n_layers,'n_heads': n_heads,
                'mlp_ratio': mlp_ratio,
            },
        },
        ckpt_dir / "encoder_final.pt",
    )
    ckpt_vol.commit()

    print(f"\nDone. Best loss: {best_loss:.4f}")
    print(f"  modal volume get iidx-ckpts {ckpt_name}/encoder_best.pt "
          f"checkpoints/{ckpt_name}/encoder_best.pt")


@app.local_entrypoint()
def main():
    train.remote()
