"""
Causal (autoregressive) event encoder for IIDX DP chart pattern learning.

Architecture
------------
Per-token input: lane_bits (16-dim float), delta_bin (int), note_type (int)
  → concat field embeddings → d_model=256
  → causal transformer (position t attends only to 0..t)
  → at position t, predict token t+1: lane_bits, delta_bin, note_type

Window embedding
----------------
encode() returns the hidden state at the last real (non-padded) token,
L2-normalised. This token has attended over the full preceding sequence
and summarises the local pattern up to the end of the window.

Loss weights (default):
  lane  1.0  ← pos_weight=5.0 to push active-bit recall
  delta 0.25
  ntype 0.50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from event_tokenize       import DELTA_BINS
from event_dataset        import NUM_LANES, NUM_NTYPES
from causal_event_dataset import MAX_TOKENS


class CausalEventEncoder(nn.Module):
    """
    Causal transformer over per-timestep event tokens.
    Use encode() at inference time to get a window embedding.
    """

    def __init__(
        self,
        d_model:   int   = 256,
        d_lane:    int   = 128,
        d_delta:   int   = 64,
        d_ntype:   int   = 64,
        n_layers:  int   = 4,
        n_heads:   int   = 8,
        mlp_ratio: float = 4.0,
        dropout:   float = 0.1,
        max_seq:   int   = MAX_TOKENS,
    ):
        super().__init__()
        if d_lane + d_delta + d_ntype != d_model:
            raise ValueError(
                f'd_lane({d_lane})+d_delta({d_delta})+d_ntype({d_ntype}) '
                f'must equal d_model({d_model})'
            )

        self.lane_proj   = nn.Linear(NUM_LANES, d_lane, bias=False)
        self.delta_embed = nn.Embedding(DELTA_BINS + 1, d_delta)
        self.ntype_embed = nn.Embedding(NUM_NTYPES + 1, d_ntype)
        self.input_norm  = nn.LayerNorm(d_model)
        self.pos_embed   = nn.Embedding(max_seq, d_model)

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
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def _embed_fields(self, lane_bits, delta_bins, note_types):
        return self.input_norm(torch.cat([
            self.lane_proj(lane_bits),
            self.delta_embed(delta_bins),
            self.ntype_embed(note_types),
        ], dim=-1))

    def forward(
        self,
        lane_bits:  torch.Tensor,  # (B, T, 16) float
        delta_bins: torch.Tensor,  # (B, T) long
        note_types: torch.Tensor,  # (B, T) long
        is_padded:  torch.Tensor,  # (B, T) bool — unused; padding always trails real tokens
    ) -> torch.Tensor:
        """Returns (B, T, d_model) causal hidden states."""
        B, T, _ = lane_bits.shape
        x   = self._embed_fields(lane_bits, delta_bins, note_types)
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        x   = x + self.pos_embed(pos)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            T, device=x.device, dtype=x.dtype)
        x = self.transformer(x, mask=causal_mask)
        return self.norm(x)

    @torch.no_grad()
    def encode(
        self,
        lane_bits:  torch.Tensor,
        delta_bins: torch.Tensor,
        note_types: torch.Tensor,
        is_padded:  torch.Tensor | None = None,
        pool:       str = 'last',
    ) -> torch.Tensor:
        """L2-normalised window embedding. Returns (B, d_model).

        pool='last' : hidden state at the last real token (default)
        pool='mean' : mean of all real-token hidden states
        """
        B, T = delta_bins.shape
        device = lane_bits.device
        if is_padded is None:
            is_padded = torch.zeros(B, T, dtype=torch.bool, device=device)
        out = self.forward(lane_bits, delta_bins, note_types, is_padded)
        if pool == 'mean':
            mask = ~is_padded                                          # (B, T)
            emb  = (out * mask.unsqueeze(-1)).sum(1)                  # (B, d)
            emb  = emb / mask.sum(1, keepdim=True).clamp(min=1)
        else:
            lengths = (~is_padded).sum(dim=1).clamp(min=1) - 1       # (B,)
            emb     = out[torch.arange(B, device=device), lengths]   # (B, d)
        return F.normalize(emb, dim=-1)


class CausalEventModel(nn.Module):
    """CausalEventEncoder + three next-token prediction heads."""

    def __init__(
        self,
        encoder:         CausalEventEncoder | None = None,
        lane_w:          float = 1.00,
        delta_w:         float = 0.25,
        ntype_w:         float = 0.50,
        lane_pos_weight: float = 5.0,
        **encoder_kwargs,
    ):
        super().__init__()
        self.encoder = encoder if encoder is not None else CausalEventEncoder(**encoder_kwargs)
        d = self.encoder.norm.normalized_shape[0]

        self.lane_head  = nn.Linear(d, NUM_LANES)
        self.delta_head = nn.Linear(d, DELTA_BINS)
        self.ntype_head = nn.Linear(d, NUM_NTYPES)

        self.lane_w  = lane_w
        self.delta_w = delta_w
        self.ntype_w = ntype_w
        self.register_buffer('lane_pos_weight', torch.full((NUM_LANES,), lane_pos_weight))

    def forward(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Returns (total_loss, metrics_dict)."""
        lane_bits  = batch['lane_bits'].float()
        delta_bins = batch['delta_bins'].long()
        note_types = batch['note_types'].long()
        is_padded  = batch['is_padded']
        tgt_lane   = batch['tgt_lane_bits'].float()
        tgt_delta  = batch['tgt_delta'].long()
        tgt_ntype  = batch['tgt_ntype'].long()
        is_valid   = batch['is_causal_valid']              # (B, T) bool

        h = self.encoder(lane_bits, delta_bins, note_types, is_padded)  # (B, T, d)

        lane_logits  = self.lane_head(h)                   # (B, T, 16)
        delta_logits = self.delta_head(h)                  # (B, T, DELTA_BINS)
        ntype_logits = self.ntype_head(h)                  # (B, T, NUM_NTYPES)

        if is_valid.any():
            lane_loss = F.binary_cross_entropy_with_logits(
                lane_logits[is_valid], tgt_lane[is_valid],
                pos_weight=self.lane_pos_weight)
        else:
            lane_loss = lane_logits.sum() * 0.0

        B, T, _ = lane_logits.shape
        delta_loss = F.cross_entropy(
            delta_logits.view(B * T, -1), tgt_delta.view(B * T), ignore_index=-100)
        ntype_loss = F.cross_entropy(
            ntype_logits.view(B * T, -1), tgt_ntype.view(B * T), ignore_index=-100)

        total = self.lane_w * lane_loss + self.delta_w * delta_loss + self.ntype_w * ntype_loss

        with torch.no_grad():
            if is_valid.any():
                pred_bits = lane_logits[is_valid] > 0
                tgt_bits  = tgt_lane[is_valid].bool()
                active    = tgt_bits
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
            'loss':        total.item(),
            'lane_loss':   lane_loss.item(),
            'delta_loss':  delta_loss.item(),
            'ntype_loss':  ntype_loss.item(),
            'lane_recall': lane_recall.item(),
            'delta_acc':   delta_acc.item(),
            'ntype_acc':   ntype_acc.item(),
        }


def model_info(model: nn.Module) -> str:
    n = sum(p.numel() for p in model.parameters())
    return f'{n / 1e6:.2f}M params'
