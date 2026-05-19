import torch
import torch.nn as nn
import torch.nn.functional as F

from mae import MAEEncoder


# ── Damage head ───────────────────────────────────────────────────────────────

class DamageHead(nn.Module):
    """
    Maps a segment embedding to per-segment judgment rate distribution.
    Output: (r, d_good, d_bad, d_poor) summing to 1 via softmax.
      r      = PGREAT + GREAT rate  → gauge recovery
      d_good = GOOD rate            → small drain (EC only)
      d_bad  = BAD + 空POOR rate    → medium drain
      d_poor = POOR rate            → heavy drain
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        # emb: (T, embed_dim)  or  (B, T, embed_dim)
        return F.softmax(self.net(emb), dim=-1)  # (..., 4)


# ── Differentiable gauge simulation ──────────────────────────────────────────

def simulate_hc(rates: torch.Tensor, note_counts: torch.Tensor) -> torch.Tensor:
    """
    Hard Clear gauge simulation (sequential, differentiable).

    rates       : (T, 4)  — (r, d_good, d_bad, d_poor) per segment
    note_counts : (T,)    — total note events per segment (int, treated as float)
    Returns     : (T,)    — gauge trajectory (0–1)
    """
    T = rates.shape[0]
    n = note_counts.float()
    r, _, d_bad, d_poor = rates[:, 0], rates[:, 1], rates[:, 2], rates[:, 3]

    recovery   = r    * n * 0.0016
    drain_bad  = d_bad  * n * 0.05
    drain_poor = d_poor * n * 0.09

    trajectory = []
    g = torch.ones(1, device=rates.device, dtype=rates.dtype)
    for t in range(T):
        # soft 30% correction: sigmoid gate ≈ 1 above 30%, ≈ 0.5 at floor
        correction = torch.sigmoid((g - 0.30) * 20.0)
        db = drain_bad[t] * (0.5 + 0.5 * correction)
        g = (g - db - drain_poor[t] + recovery[t]).clamp(0.0, 1.0)
        trajectory.append(g)

    return torch.cat(trajectory)  # (T,)


def simulate_exh(rates: torch.Tensor, note_counts: torch.Tensor) -> torch.Tensor:
    """EX-Hard gauge: no 30% correction, higher drain multipliers."""
    T = rates.shape[0]
    n = note_counts.float()
    r, _, d_bad, d_poor = rates[:, 0], rates[:, 1], rates[:, 2], rates[:, 3]

    recovery   = r    * n * 0.0016
    drain_bad  = d_bad  * n * 0.10
    drain_poor = d_poor * n * 0.18

    trajectory = []
    g = torch.ones(1, device=rates.device, dtype=rates.dtype)
    for t in range(T):
        g = (g - drain_bad[t] - drain_poor[t] + recovery[t]).clamp(0.0, 1.0)
        trajectory.append(g)

    return torch.cat(trajectory)


def simulate_ec(rates: torch.Tensor, note_counts: torch.Tensor,
                total_notes: int) -> torch.Tensor:
    """
    Easy Clear (groove) gauge simulation.
    Recovery is note-count normalised; target ≥ 0.80 to clear.
    """
    # note-count normalised recovery constant
    N = float(total_notes)
    if N < 350:
        a = (80000.0 / (N * 6)) / 50.0
    else:
        a = (80000.0 / (N * 2 + 1400.0)) / 50.0

    T = rates.shape[0]
    n = note_counts.float()
    r, d_good, d_bad, d_poor = rates[:, 0], rates[:, 1], rates[:, 2], rates[:, 3]

    recovery    = r      * n * a
    drain_good  = d_good * n * 0.016
    drain_bad   = d_bad  * n * 0.048
    drain_poor  = d_poor * n * 0.048  # POOR same as BAD for Easy gauge

    trajectory = []
    g = torch.full((1,), 0.20, device=rates.device, dtype=rates.dtype)
    for t in range(T):
        g = (g - drain_good[t] - drain_bad[t] - drain_poor[t] + recovery[t]).clamp(0.0, 1.0)
        trajectory.append(g)

    return torch.cat(trajectory)


def trajectory_summary(traj: torch.Tensor) -> torch.Tensor:
    """Compress a gauge trajectory to 4 summary scalars."""
    T = traj.shape[0]
    idx_75 = max(0, int(T * 0.75) - 1)
    idx_90 = max(0, int(T * 0.90) - 1)
    return torch.stack([
        traj.min(),
        traj.mean(),
        traj[idx_75],
        traj[-1],
        traj[idx_90],
    ])  # (5,)


# ── Rating head ───────────────────────────────────────────────────────────────

class RatingHead(nn.Module):
    """Maps gauge trajectory summary (5 scalars) → predicted decimal rating."""

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        # summary: (5,)  or  (B, 5)
        return self.net(summary).squeeze(-1)  # scalar or (B,)


# ── Full model ────────────────────────────────────────────────────────────────

class IIDXModel(nn.Module):
    """
    Full fine-tuning model.

    Processes a chart as a sequence of 4-bar windows:
      encoder   → per-segment 128-dim embedding
      damage head → (r, d_good, d_bad, d_poor) per segment
      gauge sims  → HC / EXH / EC trajectories
      rating heads → predicted (rating_hc, rating_exh, rating_ec)
    """

    def __init__(self, encoder: MAEEncoder | None = None, embed_dim: int = 128):
        super().__init__()
        self.encoder     = encoder if encoder is not None else MAEEncoder(embed_dim=embed_dim)
        self.damage_head = DamageHead(embed_dim=embed_dim)
        self.hc_head     = RatingHead()
        self.exh_head    = RatingHead()
        self.ec_head     = RatingHead()

    def forward(
        self,
        windows: torch.Tensor,      # (T, WINDOW_ROWS, IN_CHANNELS)
        note_counts: torch.Tensor,  # (T,) int32
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict with keys: 'hc', 'exh', 'ec' — predicted decimal ratings.
        """
        # encode each window independently
        embs  = self.encoder.embed_segment(windows)  # (T, embed_dim)
        rates = self.damage_head(embs)               # (T, 4)

        total_notes = int(note_counts.sum().item())

        traj_hc  = simulate_hc(rates, note_counts)
        traj_exh = simulate_exh(rates, note_counts)
        traj_ec  = simulate_ec(rates, note_counts, total_notes)

        return {
            'hc':  self.hc_head(trajectory_summary(traj_hc)),
            'exh': self.exh_head(trajectory_summary(traj_exh)),
            'ec':  self.ec_head(trajectory_summary(traj_ec)),
        }


def finetune_loss(
    pred: dict[str, torch.Tensor],
    targets: torch.Tensor,          # (3,)  [ec, hc, exh]
    weights: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> torch.Tensor:
    """MSE loss across EC, HC, EXH heads."""
    ec_w, hc_w, exh_w = weights
    return (
        ec_w  * F.mse_loss(pred['ec'],  targets[0]) +
        hc_w  * F.mse_loss(pred['hc'],  targets[1]) +
        exh_w * F.mse_loss(pred['exh'], targets[2])
    )
