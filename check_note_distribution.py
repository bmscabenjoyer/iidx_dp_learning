import pandas as pd
from pathlib import Path

data_root = Path('/home/jysuh/projects/iidx_data')
manifest = pd.read_csv(data_root / 'labeled_manifest.csv')
manifest = manifest[manifest['status'] == 'ok']

stats = []
for lv in [10, 11, 12]:
    lv_df = manifest[manifest['level'] == lv]
    total_charts = len(lv_df)
    total_notes = lv_df['num_notes'].sum()
    avg_notes = lv_df['num_notes'].mean()
    stats.append({
        'level': lv,
        'charts': total_charts,
        'total_notes': total_notes,
        'avg_notes': avg_notes
    })

df_stats = pd.DataFrame(stats)
df_stats['note_share'] = df_stats['total_notes'] / df_stats['total_notes'].sum()
df_stats['chart_share'] = df_stats['charts'] / df_stats['charts'].sum()
print(df_stats)
