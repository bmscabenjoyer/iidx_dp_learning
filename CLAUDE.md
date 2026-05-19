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
- **EC (Easy Clear):** lenient groove gauge — PGREAT/GREAT adds proportional
  recovery, GOOD subtracts ~1.6%, BAD/空POOR ~4.8%, must reach 80% to clear.
- **HC (Hard Clear):** survival gauge — PGREAT/GREAT +0.16%, GOOD 0%,
  BAD/空POOR −5%, POOR −9%; 30% correction halves BAD/空POOR damage when gauge
  < 30%; cannot recover below 30% in the same way groove can.
- **EXH (EX-Hard):** survival gauge — PGREAT/GREAT +0.16%, GOOD 0%,
  BAD/空POOR −10%, POOR −18%; NO 30% correction.

The stat rating captures real-world aggregate difficulty, which is why it's
more informative than the official integer level.

### Gauge mechanics (verified from namu.wiki)

| Event | EC/Normal | HARD | EX HARD |
|---|---|---|---|
| PGREAT | +a | +0.16% | +0.16% |
| GREAT | +a/2 | +0.16% | +0.16% |
| GOOD | −1.6% (−2.0%) | 0% | 0% |
| BAD / 空POOR | −4.8% (−6.0%) | −5% | −10% |
| POOR | same as BAD | −9% | −18% |
| 30% correction | N/A | O (halves BAD/空POOR below 30%) | X |

EC recovery constant: `a = (80000/(notes×6))/50` if notes < 350;
`a = (80000/(notes×2+1400))/50` if notes ≥ 350.

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
- **Hard endings:** a difficult final section specifically punishes HC (player
  has no gauge buffer to spend) — the reason ABMIL (permutation-invariant) is
  wrong for this problem.

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
| `file_path` | relative path to the `.npy` file within the level directory (e.g. `charts/foo_DAC00.npy`) |
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
| 12 | `rating_ec`, `rating_hc`, `rating_exh` (float) | Three-head regression targets; use `rating_stat` for single-value eval |
| 10–11 | Ordinal / range derived from clear counts | Used for "ladder" pretraining to avoid overfitting on lv12 alone |

**Regression target distribution (lv12):**
- `rating_stat`: std ≈ 0.21, mean ≈ 12.0 — very narrow; naive mean predictor MAE ≈ 0.17
- `rating_ec` / `rating_hc` / `rating_exh`: std ≈ 2.7 / 2.1 / 1.5 — much wider; use these as primary training targets
- EXH−EC gap: mean ≈ 6.06, std ≈ 1.65 → burst proxy
- HC−EC gap: wall/density proxy

The clear counts for lv10/11 can be used to construct a soft ordinal signal:
charts with very low EC count relative to total players are harder than charts
with near-universal clears.

---

## Input representation

### BPM fill-forward
Column 16 stores BPM only at change events. Fill forward to produce a
dense `bpm[row]` array. Convert to BPM directly (divide by 100).

### Two-channel note encoding
Convert the 0–9 lane encoding to two channels per lane before feeding to the model:

- **Action channel** (channel 0): 1.0 at tap notes, CN/HCN/BSS/MSS heads, and CN/HCN
  tail rows. Represents "motor event required."
- **Sustain channel** (channel 1): 1.0 during CN body, 2.0 during HCN body,
  0.5 during BSS/MSS body. Represents "hold currently required."

Input tensor shape: `(total_rows, 16, 2)` per chart — 16 lanes × 2 channels.
With BPM as a separate scalar channel per row: `(total_rows, 16×2 + 1)`.

### Gaussian note smearing
Apply a BPM-aware Gaussian blur to the action channel only (not sustain bodies):

```
σ(bpm) ≈ 0.022 × (bpm / 150)   [in seconds, GREAT window ≈ ±33ms]
σ_rows  = σ(bpm) × (bpm/60 × 48)   # convert to row units
```

At 150 BPM: σ ≈ 4 rows; at 200 BPM: σ ≈ 5 rows.
Smearing gives the model soft look-ahead context reflecting human timing windows.
Do NOT smear CN/HCN/BSS body rows (sustain channel).

### Windowing
Slice each chart into overlapping **4-bar segments** (4 × 192 = 768 rows).
Suggested stride: 2 bars (384 rows). Store note count per segment for gauge sim.

---

## Model architecture

### Overview

```
Chart (sequence of T 4-bar segments)
         │
         ▼
  CNN Encoder (shared, pretrained)
  Input: (768, 33) per segment   [16 lanes × 2 channels + 1 BPM]
         │
         │  [e₁, e₂, ..., eₜ]  128-dim per segment
         │
    ┌────┴──────────┬──────────────┐
    ▼               ▼              ▼
  HC damage head  EXH damage head  EC damage head
  outputs per segment:
    r[t]       = recovery rate (fraction of notes that are PGREAT/GREAT)
    d_bad[t]   = bad/空poor rate
    d_poor[t]  = poor rate
         │               │              │
         ▼               ▼              ▼
  HC simulation    EXH simulation   EC simulation
  (sequential,     (no 30% corr.)   (groove recovery)
   30% corr.)
         │               │              │
         ▼               ▼              ▼
  HC trajectory   EXH trajectory   EC trajectory
         │               │              │
         └───────┬────────┘              │
                 ▼                       ▼
           Gauge profile          EC rating head
           features               (MLP → rating_ec)
           [min, final, p75, p90]
                 │
                 ▼
          HC rating head     EXH rating head
          (MLP → rating_hc)  (MLP → rating_exh)

  Combined loss: MSE on rating_ec + rating_hc + rating_exh
  Eval: rating_stat = f(ec, hc, exh) — or predict directly
```

### CNN encoder
- Input: `(768, 33)` per 4-bar window — flatten to `(768, 33)` or treat as
  `(33, 768)` image-like
- First conv kernel should span all 16 lanes to capture cross-hand patterns:
  `kernel=(1, 16)` then temporal convolutions on top
- Target: 128-dim segment embedding

### Damage heads
Each of the three heads is a small MLP on top of the segment embedding.
Outputs must be in [0, 1] (use sigmoid). Interpretation:
- `r[t]`: fraction of note events in segment that yield PGREAT/GREAT recovery
- `d_bad[t]`: fraction of note events that yield BAD/空POOR
- `d_poor[t]`: fraction of note events that yield POOR (real miss)

Note count per segment `n[t]` is computed from the chart data (not learned).

### Differentiable gauge simulation

HC simulation (pseudocode):
```python
g = 1.0
for t in range(T):
    n = note_count[t]
    recovery   = r[t]    * n * 0.0016
    drain_bad  = d_bad[t] * n * 0.05
    drain_poor = d_poor[t] * n * 0.09
    # soft 30% correction gate (differentiable)
    correction = sigmoid((g - 0.30) * 20)          # ≈1 above 30%, ≈0.5 at floor
    drain_bad  = drain_bad * (0.5 + 0.5 * correction)
    g = clip(g - drain_bad - drain_poor + recovery, 0.0, 1.0)
    trajectory.append(g)
```

EXH simulation: same but no 30% correction; drain_bad multiplier = 0.10,
drain_poor multiplier = 0.18.

EC simulation: groove recovery model — recovery proportional to r[t] × a(n),
drain proportional to d_bad[t] × 0.048; clip to [0, 1]; target ≥ 0.80 to clear.

The simulation has **no learned parameters** — only r[t]/d_bad[t]/d_poor[t]
from the damage heads are learned. All arithmetic is differentiable; use
soft-min instead of hard min for gradient flow through the minimum operation.

### Why not ABMIL
Attention-Based MIL treats segments as an unordered bag — permutation invariant.
This is wrong for IIDX: a hard ending is specifically punishing for HC because
the gauge has no buffer to spend (see Go Beyond!! EC=1.1, HC=11.4, gap=10.3).
The gauge simulation enforces causal temporal structure that ABMIL cannot capture.

---

## Training strategy

### Baseline first
Before the CNN, build a GBT (XGBoost/LightGBM) baseline on handcrafted features:
- Note count, chord density (mean / max / std per bar)
- Scratch lane activity (lanes 0 and 15)
- BPM statistics
- Cross-hand note fraction
This is not a throwaway — it may be competitive given the small labeled dataset
(684 lv12 charts) and provides a MAE floor to beat.

### Self-supervised pretraining (BYOL)
Pretrain the CNN encoder on all segments from all levels (lv10/11/12) with no
labels required — 50K+ segments available. Use BYOL (Bootstrap Your Own Latent)
rather than SimCLR; BYOL works well on small batches without negative pairs.

Augmentations for contrastive views:
- **Mirror:** reverse key order within each side independently (key 1↔7, 2↔6, 3↔5, 4 stays)
- **Side-swap (FLIP):** swap P1 and P2 sides entirely (lanes 0–7 ↔ 8–15)
- Temporal crop: random 3-bar sub-window within 4-bar segment
- Gaussian noise on action channel values

### Fine-tuning
After pretraining, fine-tune the encoder + damage heads + simulation on lv12
labeled data. Use the three gauge ratings (EC/HC/EXH) as multi-task targets.

For lv10/11: auxiliary pairwise ranking loss on clear counts
(e.g. if chart A has much lower EC rate than chart B at same level, A > B in difficulty).

### Loss
```
L = MSE(rating_ec) + MSE(rating_hc) + MSE(rating_exh)
  + λ_rank × pairwise_rank_loss(lv10/11)
```

### Evaluation
- Primary: **MAE** on lv12 test set `rating_stat`
- Secondary: Spearman ρ on lv12 `rating_stat`
- Sanity: predicted mean difficulty lv10 < lv11 < lv12

---

## Augmentation

IIDX DP charts have natural symmetries:
- **Mirror (flip):** Reverse all columns within each side independently
  (key 1↔7, 2↔6, 3↔5, 4 stays). Preserves physical ergonomics. Lanes: P1 1↔6, 2↔5, 3↔4; P2 8↔13, 9↔12, 10↔11.
- **Side-swap (FLIP mode):** Swap P1 and P2 sides entirely (lanes 0–7 ↔ 8–15,
  swapping scratch positions). Also a legal chart variant in-game.

These two augmentations can 4× the effective dataset size.

---

## Pattern archetype discovery (sub-goal)

After pretraining, run K-means on the 128-dim segment embeddings from all charts.
Each cluster centre is a "pattern archetype" (e.g. scratch wall, symmetric trill,
chord stream). Assign each segment to its nearest archetype.

Per-chart archetype histogram: fraction of segments belonging to each cluster.
This histogram is a compact chart fingerprint for similarity search.

---

## Upstream data pipeline

Two source repos feed this project — do not edit data files here:

| Repo | Purpose | Output |
|---|---|---|
| `~/projects/textagextractor` | Scrapes chart data from textage.cc | `~/projects/iidx_data/dp{10,11,12}_active/` |
| `~/projects/ereterextractor` | Scrapes difficulty stats from ereter.net, merges with textage manifests | `~/projects/iidx_data/labeled_manifest.csv` |

To refresh data: run `python -m textagextractor --type dp --level N --active-only --format numpy` for each level, then `python merge.py --data-dir ~/projects/iidx_data` from ereterextractor.
