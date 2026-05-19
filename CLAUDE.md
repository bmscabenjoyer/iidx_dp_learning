# CLAUDE.md — iidx_dp_learning

This repo builds a difficulty-estimation and pattern-analysis model for
beatmania IIDX Double Play (DP) charts using data extracted from textage.cc
and labeled with community difficulty ratings from ereter.net.

---

## Project goals

### Primary goal
Predict granular decimal difficulty ratings (e.g. 12.3, 12.8) for DP charts,
beyond the official integer levels (10/11/12). Ground truth is the statistical
rating published by ereter.net, which is derived from real player clear rates
across three gauge types (EC / HC / EXH).

### Sub-goals
- **Segment contribution:** Quantify per-bar difficulty contribution so hard
  sections can be identified within a chart.
- **Chart embeddings:** Produce a latent "fingerprint" per chart for similarity
  search (find ergonomically similar charts across levels).
- **Archetype discovery:** Cluster embeddings to surface recurring pattern
  types ("scratch walls", "symmetric trills", "chord streams", etc.).
- **Explainability:** Use attention or saliency maps to show which bars drive
  the difficulty estimate.

---

## IIDX DP gameplay fundamentals

Understanding these is essential for making sensible modelling choices.

### Layout
DP uses two 7-key + scratch panels simultaneously — 16 lanes total:
- Lanes 0–7: P1 side (scratch=0, keys 1–7)
- Lanes 8–14: P2 side (keys 1–7)
- Lane 15: P2 scratch (far right)

### Note types
| Value in array | Meaning |
|---|---|
| 0 | empty |
| 1 | tap note |
| 2 | CN (charge note) head |
| 3 | CN body |
| 4 | HCN (hell charge note) head |
| 5 | HCN body |
| 6 | BSS (back-spin scratch) head |
| 7 | BSS body |
| 8 | MSS (music speed scratch) head |
| 9 | MSS body |

CN/HCN must be held until the tail; releasing early breaks the combo (HCN
immediately kills the gauge). BSS/MSS are scratch-lane variants. These special
note types are rare — most charts contain only taps and CNs.

### Gauge types and their difficulty relationship
Ereter publishes three separate statistical ratings:
- **EC (Easy Clear):** lenient gauge, forgiving recovery — lower effective
  difficulty, most players clear first.
- **HC (Hard Clear):** gauge falls fast and cannot recover below 30% — punishes
  walls and dense runs heavily.
- **EXH (EX-Hard):** any miss kills the gauge immediately — near-perfect play
  required; strongly correlated with burst density.

The stat rating captures real-world aggregate difficulty, which is why it's
more informative than the official integer level.

### What makes a chart hard (DP-specific)
- **Cross-hand patterns:** notes requiring the right hand to play on the P1
  side or vice versa.
- **Scratch-key walls:** simultaneous scratch + key notes (especially double
  scratch); lane 0 and lane 15 activity alongside dense keys.
- **Symmetric vs. asymmetric:** symmetric patterns (mirror-image between P1
  and P2) are easier to read; asymmetric ones demand independent hand tracking.
- **Chord density / simultaneity:** many lanes active in the same row.
- **BPM:** high BPM multiplies every other difficulty factor; BPM changes
  (especially sudden drops/jumps) add reading difficulty independently.
- **Note count vs. chart length:** stamina charts have sustained density; burst
  charts have local peaks.

---

## Data schema

### Numpy chart arrays
Location: `~/projects/iidx_data/dp{10,11,12}_active/charts/<id>.npy`

Shape: `(total_rows, 17)` where `total_rows = total_bars × 192`.

Column layout:
- Columns 0–15: lanes (see layout above), values 0–9 (note type encoding)
- Column 16: BPM × 100 as uint32 (0 = no change, non-zero = BPM at that row)

`ROWS_PER_BAR = 192` (4 beats × 48 subdivisions per beat). A row represents
1/192 of a bar, i.e. 1/48 of a beat.

### JSON sidecars
Location: `~/projects/iidx_data/dp{10,11,12}_active/charts/<id>.json`

Fields:
```json
{
  "title":      "song name",
  "chart_type": "DP",
  "diftype":    "[DP ANOTHER]",
  "level":      12,
  "bpm":        "184",
  "total_bars": 148,
  "hcn":        false,
  "has_bss":    false,
  "has_mss":    false
}
```

`bpm` is a string — can be a range like `"155~175"` (lower bound is used as
starting BPM in the converter). `hcn`/`has_bss`/`has_mss` are chart-level
booleans signalling rare note type presence.

### Manifest
Location: `~/projects/iidx_data/dp{10,11,12}_active/manifest.csv`

Columns: `id, title, diftype, level, status, url, ...`
`id` is the stem of the `.npy` / `.json` filenames.

### Labeled manifest
Location: `~/projects/iidx_data/labeled_manifest.csv`

The merged output from `ereterextractor/merge.py`. Every row is one textage
chart with ereter stats joined on (title, diff_type, level). Key columns:

| Column | Description |
|---|---|
| `id` | chart file stem |
| `title` | song title |
| `diftype` | e.g. `[DP ANOTHER]` |
| `level` | 10, 11, or 12 |
| `rating_stat` | **primary label** — ereter decimal stat rating (lv12 only) |
| `rating_ec` / `rating_hc` / `rating_exh` | per-gauge decimal ratings (lv12 only) |
| `count_ec` / `count_hc` / `count_exh` / `count_fc` | clear counts (all levels) |
| `count_aaa` / `count_aa` / `count_a` | score rank counts (all levels) |
| `max_score_pct` | highest recorded score % (all levels) |
| `ereter_url` | source URL (lv12 only; null for lv10/11) |

**Coverage:** 2283 / 2387 charts have ereter stats (95.6%). 684 / 731 lv12
charts have decimal ratings (93.6%). lv10/11 have counts but no decimal rating.

---

## Label scheme

| Level | Label type | Notes |
|---|---|---|
| 12 | `rating_stat` (float, ~10.0–13.0 range) | Primary regression target |
| 10–11 | Ordinal / range derived from clear counts | Used for "ladder" pretraining to avoid overfitting on lv12 alone |

The clear counts for lv10/11 can be used to construct a soft ordinal signal:
charts with very low EC count relative to total players are harder than charts
with near-universal clears.

---

## Modelling approach

### Input representation
Full charts are too long (often 100–300+ bars) for direct processing. The plan
is to window into fixed-size segments:

1. **Windowing:** Slice each chart into overlapping **4-bar segments**
   (4 × 192 = 768 rows). Stride TBD (2-bar stride suggested).
2. **Optional re-representation:** Convert the 0–9 encoding to a two-channel
   form before feeding to the model:
   - *Action channel*: 1 at every head/tail event (tap, CN head/tail, etc.)
   - *Sustain channel*: non-zero during hold bodies, with magnitude encoding
     strictness (HCN > CN, BSS/MSS > regular)
   Temporal Gaussian decay on the action channel only can give the model soft
   look-ahead context without blurring hold structure.
3. **BPM feature:** Include the BPM column (or derived tempo in BPM directly)
   as an additional input channel.

### Architecture sketch
```
4-bar window (768 × 17)
        ↓
  CNN encoder (or small Transformer)
        ↓
  128-dim segment embedding
        ↓  (repeated for all windows in a chart)
  Aggregator:
    - MaxPool  → captures peak/wall difficulty  (→ HC/EXH proxy)
    - MeanPool → captures sustained density     (→ EC proxy)
        ↓
  Song-level embedding (256-dim concat or learned mix)
        ↓
  Regression head → decimal rating
  (+ optional per-gauge heads for EC / HC / EXH)
```

Multi-task learning over the three gauge ratings is attractive because EC/HC/EXH
provide three correlated but distinct difficulty signals from the same chart.

### Loss function
- **lv12:** MSE (or Huber) on decimal rating.
- **lv10/11:** Range loss or pairwise ranking loss derived from clear counts.
- Combined weighted loss during joint training.

### Augmentation
IIDX DP charts have natural symmetries that should be exploited:
- **Mirror (flip):** Reverse all columns within each side independently
  (key 1↔7, 2↔6, 3↔5, 4 stays). Preserves physical ergonomics.
- **Side-swap (FLIP mode):** Swap P1 and P2 sides entirely (lanes 0–7 ↔ 8–15,
  swapping scratch positions). Also a legal chart variant in-game.
These two augmentations can 4× the effective dataset size.

### Evaluation
- Primary: **MAE** on lv12 test set decimal ratings.
- Secondary: Spearman rank correlation on lv12.
- Sanity check: the model should respect ordinal level ordering
  (predicted difficulty: lv10 < lv11 < lv12 on average).

---

## Upstream data pipeline

Two source repos feed this project — do not edit data files here:

| Repo | Purpose | Output |
|---|---|---|
| `~/projects/textagextractor` | Scrapes chart data from textage.cc | `~/projects/iidx_data/dp{10,11,12}_active/` |
| `~/projects/ereterextractor` | Scrapes difficulty stats from ereter.net, merges with textage manifests | `~/projects/iidx_data/labeled_manifest.csv` |

To refresh data: run `python -m textagextractor --type dp --level N --active-only --format numpy` for each level, then `python merge.py --data-dir ~/projects/iidx_data` from ereterextractor.
