"""
Modal training wrapper for IIDX DP MAE pretraining.

Setup (one-time):
    pip install modal
    modal setup                         # authenticate
    modal volume create iidx-data
    modal volume create iidx-ckpts
    modal volume put iidx-data ~/projects/iidx_data/ /

Run half-chart encoder (default, bar-based):
    modal run modal_pretrain.py

Run v7 BPM-agnostic time-based encoder:
    modal run modal_pretrain.py::main_v7

Run v8 time-based encoder (embed_dim=384, depth=4, uniform masking):
    modal run modal_pretrain.py::main_v8

Run v9 encoder (v8 + RoPE attention + per-patch BPM):
    modal run modal_pretrain.py::main_v9

Run v10 encoder (4s windows, 2s stride, embed_dim=192, RoPE):
    modal run modal_pretrain.py::main_v10

Run v11 encoder (v10 without BPM conditioning):
    modal run modal_pretrain.py::main_v11

Run v12 encoder (4-row patches, 33ms, 75% mask):
    modal run modal_pretrain.py::main_v12

Run v6 full-chart encoder:
    modal run modal_pretrain.py::main_v6

Download checkpoint after training:
    modal volume get iidx-ckpts pretrain_half_v8/encoder_best.pt checkpoints/pretrain_half_v8/encoder_best.pt
    modal volume get iidx-ckpts pretrain_half_v7/encoder_best.pt checkpoints/pretrain_half_v7/encoder_best.pt
    modal volume get iidx-ckpts pretrain_half/encoder_best.pt    checkpoints/pretrain_half/encoder_best.pt
    modal volume get iidx-ckpts pretrain_v6/encoder_best.pt      checkpoints/pretrain_v6/encoder_best.pt
"""

import modal

# ── Image ─────────────────────────────────────────────────────────────────────

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "pandas", "tqdm", "scipy", "scikit-learn")
    .pip_install("torch==2.5.1",
                 index_url="https://download.pytorch.org/whl/cu124")
    .add_local_file("pretrain.py", "/app/pretrain.py")
    .add_local_file("mae.py",      "/app/mae.py")
    .add_local_file("dataset.py",  "/app/dataset.py")
)

# ── Volumes ───────────────────────────────────────────────────────────────────

data_vol = modal.Volume.from_name("iidx-data")
ckpt_vol = modal.Volume.from_name("iidx-ckpts", create_if_missing=True)

DATA_DIR = "/iidx-data"
CKPT_DIR = "/iidx-ckpts"

# ── App ───────────────────────────────────────────────────────────────────────

app = modal.App("iidx-dp-pretrain")

# ── Half-chart encoder (8-lane, P1-format) ────────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=86400,
)
def train_half(
    epochs: int = 150,
    batch_size: int = 1024,
    lr: float = 2e-3,
    mask_ratio: float = 0.50,
    event_weight: float = 100.0,
    embed_dim: int = 256,
    decoder_dim: int = 128,
    encoder_depth: int = 6,
    decoder_depth: int = 2,
    num_heads: int = 8,
    workers: int = 4,
    compile_model: bool = True,
):
    import subprocess, sys

    cmd = [
        sys.executable, "/app/pretrain.py",
        "--manifest",      f"{DATA_DIR}/labeled_manifest.csv",
        "--data-root",     DATA_DIR,
        "--out-dir",       f"{CKPT_DIR}/pretrain_half",
        "--half",
        "--mask-ratio",    str(mask_ratio),
        "--event-weight",  str(event_weight),
        "--empty-weight",  "1.0",
        "--batch-size",    str(batch_size),
        "--lr",            str(lr),
        "--epochs",        str(epochs),
        "--embed-dim",     str(embed_dim),
        "--decoder-dim",   str(decoder_dim),
        "--encoder-depth", str(encoder_depth),
        "--decoder-depth", str(decoder_depth),
        "--num-heads",     str(num_heads),
        "--workers",       str(workers),
    ]
    if compile_model:
        cmd.append("--compile")

    result = subprocess.run(cmd, cwd="/app", check=True)
    ckpt_vol.commit()
    return result.returncode


# ── v6 full-chart encoder (16-lane, for reference) ────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=86400,
)
def train_v6(
    epochs: int = 100,
    batch_size: int = 2048,
    lr: float = 4e-3,
    mask_ratio: float = 0.6,
    event_weight: float = 100.0,
    embed_dim: int = 256,
    decoder_dim: int = 128,
    encoder_depth: int = 6,
    num_heads: int = 8,
    workers: int = 4,
    compile_model: bool = True,
):
    import subprocess, sys

    cmd = [
        sys.executable, "/app/pretrain.py",
        "--manifest",      f"{DATA_DIR}/labeled_manifest.csv",
        "--data-root",     DATA_DIR,
        "--out-dir",       f"{CKPT_DIR}/pretrain_v6",
        "--v6",
        "--mask-ratio",    str(mask_ratio),
        "--event-weight",  str(event_weight),
        "--empty-weight",  "1.0",
        "--batch-size",    str(batch_size),
        "--lr",            str(lr),
        "--epochs",        str(epochs),
        "--embed-dim",     str(embed_dim),
        "--decoder-dim",   str(decoder_dim),
        "--encoder-depth", str(encoder_depth),
        "--workers",       str(workers),
    ]
    if compile_model:
        cmd.append("--compile")

    result = subprocess.run(cmd, cwd="/app", check=True)
    ckpt_vol.commit()
    return result.returncode


# ── Local entrypoints ─────────────────────────────────────────────────────────

@app.local_entrypoint()
def main():
    """Default: half-chart encoder pretraining."""
    print("Submitting half-chart pretraining job to Modal (A10G) ...")
    ret = train_half.remote()
    print(f"Done. Return code: {ret}")
    print("Download checkpoint with:")
    print("  modal volume get iidx-ckpts pretrain_half/encoder_best.pt checkpoints/pretrain_half/encoder_best.pt")


@app.function(
    image=image,
    gpu="A10G",
    volumes={
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=86400,
)
def train_half_v7(
    epochs: int = 150,
    batch_size: int = 1024,
    lr: float = 2e-3,
    mask_ratio: float = 0.50,
    event_weight: float = 100.0,
    embed_dim: int = 256,
    decoder_dim: int = 128,
    encoder_depth: int = 6,
    decoder_depth: int = 2,
    num_heads: int = 8,
    workers: int = 4,
    compile_model: bool = True,
):
    import subprocess, sys

    cmd = [
        sys.executable, "/app/pretrain.py",
        "--manifest",      f"{DATA_DIR}/labeled_manifest.csv",
        "--data-root",     DATA_DIR,
        "--out-dir",       f"{CKPT_DIR}/pretrain_half_v7",
        "--half-v7",
        "--mask-ratio",    str(mask_ratio),
        "--event-weight",  str(event_weight),
        "--empty-weight",  "1.0",
        "--batch-size",    str(batch_size),
        "--lr",            str(lr),
        "--epochs",        str(epochs),
        "--embed-dim",     str(embed_dim),
        "--decoder-dim",   str(decoder_dim),
        "--encoder-depth", str(encoder_depth),
        "--decoder-depth", str(decoder_depth),
        "--num-heads",     str(num_heads),
        "--workers",       str(workers),
    ]
    if compile_model:
        cmd.append("--compile")

    result = subprocess.run(cmd, cwd="/app", check=True)
    ckpt_vol.commit()
    return result.returncode


@app.local_entrypoint()
def main_v7():
    """BPM-agnostic time-based half-chart encoder pretraining (v7)."""
    print("Submitting half-chart v7 pretraining job to Modal (A10G) ...")
    ret = train_half_v7.remote()
    print(f"Done. Return code: {ret}")
    print("Download checkpoint with:")
    print("  modal volume get iidx-ckpts pretrain_half_v7/encoder_best.pt checkpoints/pretrain_half_v7/encoder_best.pt")


@app.local_entrypoint()
def main_v6():
    """Full 16-lane v6 encoder pretraining."""
    print("Submitting v6 pretraining job to Modal (A10G) ...")
    ret = train_v6.remote()
    print(f"Done. Return code: {ret}")
    print("Download checkpoint with:")
    print("  modal volume get iidx-ckpts pretrain_v6/encoder_best.pt checkpoints/pretrain_v6/encoder_best.pt")


# ── v8: time-based windows, embed_dim=384, depth=4, uniform masking ───────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=86400,
)
def train_half_v8(
    epochs: int = 150,
    batch_size: int = 1024,
    lr: float = 2e-3,
    mask_ratio: float = 0.50,
    event_weight: float = 100.0,
    embed_dim: int = 384,
    decoder_dim: int = 128,
    encoder_depth: int = 4,
    decoder_depth: int = 2,
    num_heads: int = 6,
    workers: int = 4,
    compile_model: bool = True,
):
    import subprocess, sys

    cmd = [
        sys.executable, "/app/pretrain.py",
        "--manifest",      f"{DATA_DIR}/labeled_manifest.csv",
        "--data-root",     DATA_DIR,
        "--out-dir",       f"{CKPT_DIR}/pretrain_half_v8",
        "--half-v8",
        "--mask-ratio",    str(mask_ratio),
        "--event-weight",  str(event_weight),
        "--empty-weight",  "1.0",
        "--batch-size",    str(batch_size),
        "--lr",            str(lr),
        "--epochs",        str(epochs),
        "--embed-dim",     str(embed_dim),
        "--decoder-dim",   str(decoder_dim),
        "--encoder-depth", str(encoder_depth),
        "--decoder-depth", str(decoder_depth),
        "--num-heads",     str(num_heads),
        "--workers",       str(workers),
    ]
    if compile_model:
        cmd.append("--compile")

    result = subprocess.run(cmd, cwd="/app", check=True)
    ckpt_vol.commit()
    return result.returncode


@app.local_entrypoint()
def main_v8():
    """v8: time-based windows, embed_dim=384, depth=4, uniform masking, event_weight=100."""
    print("Submitting half-chart v8 pretraining job to Modal (A10G) ...")
    ret = train_half_v8.remote()
    print(f"Done. Return code: {ret}")
    print("Download checkpoint with:")
    print("  modal volume get iidx-ckpts pretrain_half_v8/encoder_best.pt checkpoints/pretrain_half_v8/encoder_best.pt")


# ── v9: v8 + RoPE attention + per-patch BPM ───────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=86400,
)
def train_half_v9(
    epochs: int = 150,
    batch_size: int = 1024,
    lr: float = 2e-3,
    mask_ratio: float = 0.50,
    event_weight: float = 100.0,
    embed_dim: int = 384,
    decoder_dim: int = 128,
    encoder_depth: int = 4,
    decoder_depth: int = 2,
    num_heads: int = 6,
    workers: int = 4,
    compile_model: bool = True,
):
    import subprocess, sys

    cmd = [
        sys.executable, "/app/pretrain.py",
        "--manifest",      f"{DATA_DIR}/labeled_manifest.csv",
        "--data-root",     DATA_DIR,
        "--out-dir",       f"{CKPT_DIR}/pretrain_half_v9",
        "--half-v8",       # same dataset as v8
        "--rope",          # RoPE attention + per-patch BPM
        "--mask-ratio",    str(mask_ratio),
        "--event-weight",  str(event_weight),
        "--empty-weight",  "1.0",
        "--batch-size",    str(batch_size),
        "--lr",            str(lr),
        "--epochs",        str(epochs),
        "--embed-dim",     str(embed_dim),
        "--decoder-dim",   str(decoder_dim),
        "--encoder-depth", str(encoder_depth),
        "--decoder-depth", str(decoder_depth),
        "--num-heads",     str(num_heads),
        "--workers",       str(workers),
    ]
    if compile_model:
        cmd.append("--compile")

    result = subprocess.run(cmd, cwd="/app", check=True)
    ckpt_vol.commit()
    return result.returncode


@app.local_entrypoint()
def main_v9():
    """v9: v8 + RoPE attention (no absolute PE) + per-patch BPM conditioning."""
    print("Submitting half-chart v9 pretraining job to Modal (A10G) ...")
    ret = train_half_v9.remote()
    print(f"Done. Return code: {ret}")
    print("Download checkpoint with:")
    print("  modal volume get iidx-ckpts pretrain_half_v9/encoder_best.pt checkpoints/pretrain_half_v9/encoder_best.pt")


# ── v10: 4s windows, 2s stride, embed_dim=192, depth=4, RoPE ─────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=86400,
)
def train_half_v10(
    epochs: int = 150,
    batch_size: int = 2048,
    lr: float = 2e-3,
    mask_ratio: float = 0.50,
    event_weight: float = 100.0,
    embed_dim: int = 192,
    decoder_dim: int = 96,
    encoder_depth: int = 4,
    decoder_depth: int = 2,
    num_heads: int = 4,
    workers: int = 4,
    compile_model: bool = True,
):
    import subprocess, sys

    cmd = [
        sys.executable, "/app/pretrain.py",
        "--manifest",      f"{DATA_DIR}/labeled_manifest.csv",
        "--data-root",     DATA_DIR,
        "--out-dir",       f"{CKPT_DIR}/pretrain_half_v10",
        "--half-v10",
        "--rope",
        "--mask-ratio",    str(mask_ratio),
        "--event-weight",  str(event_weight),
        "--empty-weight",  "1.0",
        "--batch-size",    str(batch_size),
        "--lr",            str(lr),
        "--epochs",        str(epochs),
        "--embed-dim",     str(embed_dim),
        "--decoder-dim",   str(decoder_dim),
        "--encoder-depth", str(encoder_depth),
        "--decoder-depth", str(decoder_depth),
        "--num-heads",     str(num_heads),
        "--workers",       str(workers),
    ]
    if compile_model:
        cmd.append("--compile")

    result = subprocess.run(cmd, cwd="/app", check=True)
    ckpt_vol.commit()
    return result.returncode


@app.local_entrypoint()
def main_v10():
    """v10: 4s windows, 2s stride, embed_dim=192, depth=4, RoPE, 50% mask."""
    print("Submitting half-chart v10 pretraining job to Modal (A10G) ...")
    ret = train_half_v10.remote()
    print(f"Done. Return code: {ret}")
    print("Download checkpoint with:")
    print("  modal volume get iidx-ckpts pretrain_half_v10/encoder_best.pt checkpoints/pretrain_half_v10/encoder_best.pt")


# ── v11: v10 without BPM conditioning ────────────────────────────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=86400,
)
def train_half_v11(
    epochs: int = 150,
    batch_size: int = 2048,
    lr: float = 2e-3,
    mask_ratio: float = 0.50,
    event_weight: float = 100.0,
    embed_dim: int = 192,
    decoder_dim: int = 96,
    encoder_depth: int = 4,
    decoder_depth: int = 2,
    num_heads: int = 4,
    workers: int = 4,
    compile_model: bool = True,
):
    import subprocess, sys

    cmd = [
        sys.executable, "/app/pretrain.py",
        "--manifest",      f"{DATA_DIR}/labeled_manifest.csv",
        "--data-root",     DATA_DIR,
        "--out-dir",       f"{CKPT_DIR}/pretrain_half_v11",
        "--half-v10",
        "--rope",
        "--no-bpm-cond",
        "--mask-ratio",    str(mask_ratio),
        "--event-weight",  str(event_weight),
        "--empty-weight",  "1.0",
        "--batch-size",    str(batch_size),
        "--lr",            str(lr),
        "--epochs",        str(epochs),
        "--embed-dim",     str(embed_dim),
        "--decoder-dim",   str(decoder_dim),
        "--encoder-depth", str(encoder_depth),
        "--decoder-depth", str(decoder_depth),
        "--num-heads",     str(num_heads),
        "--workers",       str(workers),
    ]
    if compile_model:
        cmd.append("--compile")

    result = subprocess.run(cmd, cwd="/app", check=True)
    ckpt_vol.commit()
    return result.returncode


@app.local_entrypoint()
def main_v11():
    """v11: v10 without BPM conditioning — forces pattern structure over BPM fingerprint."""
    print("Submitting half-chart v11 pretraining job to Modal (A10G) ...")
    ret = train_half_v11.remote()
    print(f"Done. Return code: {ret}")
    print("Download checkpoint with:")
    print("  modal volume get iidx-ckpts pretrain_half_v11/encoder_best.pt checkpoints/pretrain_half_v11/encoder_best.pt")


# ── v12: 4-row patches (~33ms), 120 patches/window, 75% mask ─────────────────

@app.function(
    image=image,
    gpu="A10G",
    volumes={
        DATA_DIR: data_vol,
        CKPT_DIR: ckpt_vol,
    },
    timeout=86400,
)
def train_half_v12(
    epochs: int = 150,
    batch_size: int = 1024,
    lr: float = 2e-3,
    mask_ratio: float = 0.75,
    event_weight: float = 100.0,
    embed_dim: int = 192,
    decoder_dim: int = 96,
    encoder_depth: int = 4,
    decoder_depth: int = 2,
    num_heads: int = 4,
    workers: int = 4,
    compile_model: bool = True,
):
    import subprocess, sys

    cmd = [
        sys.executable, "/app/pretrain.py",
        "--manifest",      f"{DATA_DIR}/labeled_manifest.csv",
        "--data-root",     DATA_DIR,
        "--out-dir",       f"{CKPT_DIR}/pretrain_half_v12",
        "--half-v12",
        "--rope",
        "--no-bpm-cond",
        "--mask-ratio",    str(mask_ratio),
        "--event-weight",  str(event_weight),
        "--empty-weight",  "1.0",
        "--batch-size",    str(batch_size),
        "--lr",            str(lr),
        "--epochs",        str(epochs),
        "--embed-dim",     str(embed_dim),
        "--decoder-dim",   str(decoder_dim),
        "--encoder-depth", str(encoder_depth),
        "--decoder-depth", str(decoder_depth),
        "--num-heads",     str(num_heads),
        "--workers",       str(workers),
    ]
    if compile_model:
        cmd.append("--compile")

    result = subprocess.run(cmd, cwd="/app", check=True)
    ckpt_vol.commit()
    return result.returncode


@app.local_entrypoint()
def main_v12():
    """v12: 4-row patches (33ms), 120 patches/window, 75% mask — resolves 32nd notes."""
    print("Submitting half-chart v12 pretraining job to Modal (A10G) ...")
    ret = train_half_v12.remote()
    print(f"Done. Return code: {ret}")
    print("Download checkpoint with:")
    print("  modal volume get iidx-ckpts pretrain_half_v12/encoder_best.pt checkpoints/pretrain_half_v12/encoder_best.pt")
