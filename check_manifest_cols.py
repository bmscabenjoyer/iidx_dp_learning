import pandas as pd
import numpy as np
from pathlib import Path
import json

data_root = Path('/home/jysuh/projects/iidx_data')
manifest = pd.read_csv(data_root / 'labeled_manifest.csv')

def get_stats(level):
    lv_data = manifest[manifest['level'] == level]
    counts = []
    for _, row in lv_data.iterrows():
        npy = data_root / f'dp{level}_active' / str(row['file_path'])
        if npy.exists():
            # Load the note counts from JSON if possible, otherwise we'd need to load NPY
            # But manifest might have some count info. Let's check manifest columns.
            pass
    # Let's just look at the manifest columns first.
    return manifest.columns

print(list(manifest.columns))
