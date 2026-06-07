"""
Event token encoder and MAE wrapper for IIDX DP chart pattern pretraining.

Architecture
------------
Per-token input: lane_bits (16-dim float), delta_bin (int), note_type (int)
  → concat field embeddings → d_model
  → [CLS, tok_0 … tok_{T-1}] through transformer (with absolute PE)
  → CLS output = window embedding for similarity search

MAE objective
-------------
Masked positions → replaced with learned mask_token before transformer.
Three reconstruction heads:
  lane_head  → 16-dim binary   (BCEWithLogitsLoss, masked positions only)
  delta_head → DELTA_BINS-way  (CrossEntropyLoss,  ignore_index=-100)
  ntype_head → 4-way           (CrossEntropyLoss,  ignore_index=-100)

Loss weights (default):
  lane  1.0  ← most important: encodes chord / pattern structure
  delta 0.25 ← physical timing, secondary
  ntype 0.50 ← tap/CN/HCN, tertiary
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from event_tokenize import DELTA_BINS
from event_dataset   import NUM_LANES, NUM_NTYPES, MAX_TOKENS


class EventEncoder(nn.Module):
    """
    Transformer encoder over per-timestep event tokens.

    Use encode() at inference time to get the CLS window embedding.
    """

    def __init__(
        self,
        d_model:   int   = 128,
        d_lane:    int   = 64,
        d_delta:   int   = 32,
        d_ntype:   int   = 32,
        n_layers:  int   = 4,
        n_heads:   int   = 4,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.1,
        max_seq:   int   = MAX_TOKENS + 1,  # +1 for CLS
    ):
        super().__init__()
        if d_lane + d_delta + d_ntype != d_model:
            raise ValueError(
                f'd_lane({d_lane})+d_delta({d_delta})+d_ntype({d_ntype}) '
                f'must equal d_model({d_model})'
            )

        # Field embeddings
        self.lane_proj   = nn.Linear(NUM_LANES, d_lane, bias=False)
        self.delta_embed = nn.Embedding(DELTA_BINS + 1, d_delta)  # +1 for pad
        self.ntype_embed = nn.Embedding(NUM_NTYPES + 1, d_ntype)  # +1 for pad
        self.input_norm  = nn.LayerNorm(d_model)

        # Sequence tokens
        self.pos_embed  = nn.Embedding(max_seq, d_model)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token,  std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _embed_fields(
        self,
        lane_bits:  torch.Tensor,  # (B, T, 16)
        delta_bins: torch.Tensor,  # (B, T)
        note_types: torch.Tensor,  # (B, T)
    ) -> torch.Tensor:
        return self.input_norm(torch.cat([
            self.lane_proj(lane_bits),
            self.delta_embed(delta_bins),
            self.ntype_embed(note_types),
        ], dim=-1))                                # (B, T, d_model)

    def forward(
        self,
        lane_bits:  torch.Tensor,  # (B, T, 16) float
        delta_bins: torch.Tensor,  # (B, T) long
        note_types: torch.Tensor,  # (B, T) long
        is_masked:  torch.Tensor,  # (B, T) bool — positions to hide
        is_padded:  torch.Tensor,  # (B, T) bool — positions to ignore
    ) -> torch.Tensor:
        """Returns (B, T+1, d_model). Position 0 is CLS."""
        B, T, _ = lane_bits.shape

        x = self._embed_fields(lane_bits, delta_bins, note_types)  # (B, T, d)

        # Replace masked positions with mask token
        x = torch.where(
            is_masked.unsqueeze(-1),
            self.mask_token.expand(B, T, -1),
            x,
        )

        # Prepend CLS and add positional embeddings
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)                           # (B, T+1, d)
        pos = torch.arange(T + 1, device=x.device).unsqueeze(0)
        x   = x + self.pos_embed(pos)

        # Padding mask: CLS is never padded
        pad_mask = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=x.device),
            is_padded,
        ], dim=1)                                                   # (B, T+1)

        x = self.transformer(x, src_key_padding_mask=pad_mask)
        return self.norm(x)

    @torch.no_grad()
    def encode(
        self,
        lane_bits:  torch.Tensor,
        delta_bins: torch.Tensor,
        note_types: torch.Tensor,
        is_padded:  torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Mean-pool over non-padded token positions. Returns (B, d_model)."""
        B, T = delta_bins.shape
        device = lane_bits.device
        no_mask = torch.zeros(B, T, dtype=torch.bool, device=device)
        if is_padded is None:
            is_padded = torch.zeros(B, T, dtype=torch.bool, device=device)
        out = self.forward(lane_bits, delta_bins, note_types, no_mask, is_padded)
        tokens = out[:, 1:]                                      # (B, T, d) skip CLS
        valid  = (~is_padded).unsqueeze(-1).float()              # (B, T, 1)
        return (tokens * valid).sum(1) / valid.sum(1).clamp(min=1)


class EventMAEModel(nn.Module):
    """
    EventEncoder + three reconstruction heads for MAE pretraining.
    """

    def __init__(
        self,
        encoder:        EventEncoder | None = None,
        lane_w:         float = 1.00,
        delta_w:        float = 0.25,
        ntype_w:        float = 0.50,
        lane_pos_weight: float = 5.0,
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else EventEncoder(**encoder_kwargs)
        d = self.encoder.norm.normalized_shape[0]

        self.lane_head  = nn.Linear(d, NUM_LANES)
        self.delta_head = nn.Linear(d, DELTA_BINS)
        self.ntype_head = nn.Linear(d, NUM_NTYPES)

        self.lane_w  = lane_w
        self.delta_w = delta_w
        self.ntype_w = ntype_w
        self.register_buffer(
            'lane_pos_weight',
            torch.full((NUM_LANES,), lane_pos_weight),
        )

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Returns (total_loss, metrics_dict)."""
        lane_bits  = batch['lane_bits'].float()
        delta_bins = batch['delta_bins'].long()
        note_types = batch['note_types'].long()
        is_masked  = batch['is_masked']
        is_padded  = batch['is_padded']
        tgt_lane   = batch['tgt_lane_bits'].float()
        tgt_delta  = batch['tgt_delta'].long()
        tgt_ntype  = batch['tgt_ntype'].long()

        out = self.encoder(lane_bits, delta_bins, note_types, is_masked, is_padded)
        h   = out[:, 1:]                                # (B, T, d) — skip CLS

        lane_logits  = self.lane_head(h)                # (B, T, 16)
        delta_logits = self.delta_head(h)               # (B, T, DELTA_BINS)
        ntype_logits = self.ntype_head(h)               # (B, T, NUM_NTYPES)

        # Lane: BCE only at masked positions, upweight active bits
        if is_masked.any():
            lane_loss = F.binary_cross_entropy_with_logits(
                lane_logits[is_masked], tgt_lane[is_masked],
                pos_weight=self.lane_pos_weight)
        else:
            lane_loss = lane_logits.sum() * 0.0

        # Delta + ntype: CE with -100 ignored
        B, T, _ = lane_logits.shape
        delta_loss = F.cross_entropy(
            delta_logits.view(B * T, -1), tgt_delta.view(B * T), ignore_index=-100)
        ntype_loss = F.cross_entropy(
            ntype_logits.view(B * T, -1), tgt_ntype.view(B * T), ignore_index=-100)

        total = self.lane_w * lane_loss + self.delta_w * delta_loss + self.ntype_w * ntype_loss

        with torch.no_grad():
            if is_masked.any():
                pred_bits  = lane_logits[is_masked] > 0       # (N, 16) bool
                tgt_bits   = tgt_lane[is_masked].bool()        # (N, 16) bool
                active     = tgt_bits                          # positive positions
                lane_recall = (pred_bits[active].float().mean()
                               if active.any() else torch.zeros(1).squeeze())
            else:
                lane_recall = torch.zeros(1).squeeze()

            valid_d   = tgt_delta != -100
            valid_n   = tgt_ntype != -100
            delta_acc = (delta_logits.argmax(-1)[valid_d] == tgt_delta[valid_d]
                         ).float().mean() if valid_d.any() else torch.zeros(1).squeeze()
            ntype_acc = (ntype_logits.argmax(-1)[valid_n] == tgt_ntype[valid_n]
                         ).float().mean() if valid_n.any() else torch.zeros(1).squeeze()

        return total, {
            'loss':         total.item(),
            'lane_loss':    lane_loss.item(),
            'delta_loss':   delta_loss.item(),
            'ntype_loss':   ntype_loss.item(),
            'lane_recall':  lane_recall.item(),
            'delta_acc':    delta_acc.item(),
            'ntype_acc':    ntype_acc.item(),
        }


def model_info(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f'{n / 1e6:.2f}M params'
