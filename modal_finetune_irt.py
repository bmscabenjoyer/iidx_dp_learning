"""
Modal runner for IRT-based gauge simulation fine-tuning.

Run:
    modal run modal_finetune_irt.py

Download results:
    modal volume get iidx-ckpts finetune_irt/results_irt.txt results_irt.txt
    modal volume get iidx-ckpts finetune_irt/predictions_irt.csv predictions_irt.csv
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pandas", "tqdm", "scipy", "scikit-learn")
    .pip_install("torch==2.5.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .add_local_file("finetune_irt.py", "/app/finetune_irt.py")
    .add_local_file("mae.py",          "/app/mae.py")
    .add_local_file("dataset.py",      "/app/dataset.py")
)

data_vol = modal.Volume.from_name("iidx-data")
ckpt_vol = modal.Volume.from_name("iidx-ckpts", create_if_missing=True)

DATA_DIR = "/iidx-data"
CKPT_DIR = "/iidx-ckpts"

app = modal.App("iidx-dp-finetune-irt")


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=86400,
)
def finetune_irt():
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Build the e2e_v11 window cache (idempotent — skips if already present)
    import numpy as np
    import pandas as pd
    import torch
    from pathlib import Path

    cache_dir = Path(f"{CKPT_DIR}/cache_e2e_v11")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Populate cache if needed (reuse E2E cache build logic)
    from dataset import (
        encode_chart_pretrain_v6, window_chart_time_half,
        WIN_SECS_V10, WIN_ROWS_V10,
    )
    from mae import MAEModel
    from dataset import (
        PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF,
        NUM_PATCHES_V10, PATCH_ROWS_V7,
    )
    from tqdm import tqdm

    manifest = pd.read_csv(f"{DATA_DIR}/labeled_manifest.csv")
    lv12 = manifest[
        (manifest['level'] == 12) &
        manifest['rating_ec'].notna() &
        (manifest['status'] == 'ok')
    ].reset_index(drop=True)

    missing = [i for i in range(len(lv12))
               if not (cache_dir / f'{i}.npz').exists()]

    if missing:
        print(f'building cache: {len(missing)} charts...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ckpt   = Path(f"{CKPT_DIR}/pretrain_half_v11/encoder_best.pt")
        mae    = MAEModel(
            in_channels=PRETRAIN_IN_CHANNELS_HALF,
            num_patches=NUM_PATCHES_V10, patch_rows=PATCH_ROWS_V7,
            encoder_dim=192, decoder_dim=96, num_heads=4,
            encoder_depth=4, use_rope=True,
            use_lane_type_embed=True, use_bpm_cond=False,
            lane_types=LANE_TYPES_HALF,
        )
        mae.encoder.load_state_dict(torch.load(ckpt, map_location='cpu'))

        for i in tqdm(missing):
            row = lv12.iloc[i]
            npy = Path(DATA_DIR) / 'dp12_active' / str(row['file_path'])
            if not npy.exists():
                continue
            enc, nc, _ = encode_chart_pretrain_v6(str(npy))
            p1_wins, p2_wins, win_nc = window_chart_time_half(
                enc, nc, win_secs=WIN_SECS_V10, stride_secs=WIN_SECS_V10 / 2,
                win_rows=WIN_ROWS_V10,
            )
            if p1_wins.shape[0] == 0:
                continue
            np.savez(cache_dir / f'{i}.npz',
                     p1=p1_wins[:, :, :8].astype(np.float16),
                     p2=p2_wins[:, :, :8].astype(np.float16),
                     win_nc=win_nc.astype(np.float32))
        print(f'cache complete ({len(lv12)} charts)')
    else:
        print(f'cache complete ({len(lv12)} charts)')

    # Run IRT fine-tuning
    import finetune_irt as ft
    results_dir = f"{CKPT_DIR}/finetune_irt"

    ft.main(
        manifest_path=f"{DATA_DIR}/labeled_manifest.csv",
        data_root=DATA_DIR,
        ckpt_path=f"{CKPT_DIR}/pretrain_half_v11/encoder_best.pt",
        cache_dir=str(cache_dir),
        results_dir=results_dir,
    )

    ckpt_vol.commit()
    print("Done.", flush=True)


@app.local_entrypoint()
def main():
    finetune_irt.remote()
