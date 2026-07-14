import time, torch
from torch.utils.data import DataLoader
from dataset import PretrainDataset

if __name__ == '__main__':
    t0 = time.perf_counter()
    ds = PretrainDataset(
        manifest_csv='/home/jysuh/projects/iidx_data/labeled_manifest.csv',
        data_root='/home/jysuh/projects/iidx_data',
        augment=True,
    )
    print(f"init: {time.perf_counter()-t0:.1f}s  windows: {len(ds):,}")

    item = ds[0]
    print(f"item: {item.shape}  dtype={item.dtype}  [{item.min():.3f}, {item.max():.3f}]")

    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=4,
                        pin_memory=True, prefetch_factor=4, persistent_workers=True)
    it = iter(loader)
    _ = next(it)
    t0 = time.perf_counter()
    for i, batch in enumerate(it):
        if i == 99: break
    elapsed = (time.perf_counter() - t0) / 100
    steps_per_epoch = 788_392 // 512
    print(f"loader: {elapsed*1000:.1f} ms/batch  |  {512/elapsed:,.0f} samp/s  "
          f"|  ~{steps_per_epoch * elapsed / 60:.2f} min/epoch")
