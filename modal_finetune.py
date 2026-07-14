"""
Modal wrapper for LoRA fine-tuning of the half-chart encoder.

Run:
    modal run modal_finetune.py

Download results:
    modal volume get iidx-ckpts finetune_lora/results.txt finetune_lora_results.txt
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pandas", "tqdm", "scipy", "scikit-learn")
    .pip_install("torch==2.5.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .add_local_file("finetune_lora.py", "/app/finetune_lora.py")
    .add_local_file("mae.py",           "/app/mae.py")
    .add_local_file("dataset.py",       "/app/dataset.py")
)

data_vol = modal.Volume.from_name("iidx-data")
ckpt_vol = modal.Volume.from_name("iidx-ckpts", create_if_missing=True)

DATA_DIR = "/iidx-data"
CKPT_DIR = "/iidx-ckpts"

app = modal.App("iidx-dp-finetune")


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=86400,
)
def finetune_lora():
    import sys, os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    import finetune_lora as fl
    from pathlib import Path

    # override paths to Modal volumes
    results = fl.run(
        manifest_path=f"{DATA_DIR}/labeled_manifest.csv",
        data_root=DATA_DIR,
        ckpt_path=f"{CKPT_DIR}/pretrain_half/encoder_best.pt",
    )

    # save results to volume
    import numpy as np
    out_dir = Path(f"{CKPT_DIR}/finetune_lora")
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_vol.commit()
    print("Done.", flush=True)


@app.local_entrypoint()
def main():
    finetune_lora.remote()
