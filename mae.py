import math
import torch
import torch.nn as nn

from dataset import (WINDOW_ROWS, NUM_PATCHES, PATCH_ROWS, IN_CHANNELS,
                     PRETRAIN_IN_CHANNELS, PRETRAIN_IN_CHANNELS_V3,
                     NUM_PATCHES_V4, PATCH_ROWS_V4,
                     PRETRAIN_IN_CHANNELS_HALF, LANE_TYPES_HALF)


# ── Rotary Position Embedding ─────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)          # (S, head_dim/2)
        emb = torch.cat([freqs, freqs], dim=-1)        # (S, head_dim)
        return emb.cos()[None, None], emb.sin()[None, None]  # (1,1,S,D)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, S, head_dim), cos/sin: (1, 1, S, head_dim)"""
    return x * cos + _rotate_half(x) * sin


class RoPEAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv  = nn.Linear(embed_dim, 3 * embed_dim)
        self.out  = nn.Linear(embed_dim, embed_dim)
        self.drop = nn.Dropout(dropout)
        self.rope = RotaryEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                        # each (B, H, S, head_dim)
        cos, sin = self.rope(S, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        attn = self.drop((q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1))
        return self.out((attn @ v).transpose(1, 2).reshape(B, S, D))


class RoPETransformerLayer(nn.Module):
    """Pre-norm transformer layer with RoPE self-attention."""

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.attn  = RoPEAttention(embed_dim, num_heads, dropout)
        ff_dim     = int(embed_dim * mlp_ratio)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class RoPETransformerEncoder(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, depth: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            RoPETransformerLayer(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ── PatchEmbed ────────────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Flatten each patch and project to embed_dim."""

    def __init__(self, patch_rows: int = PATCH_ROWS, in_channels: int = IN_CHANNELS,
                 embed_dim: int = 128, use_lane_type_embed: bool = False,
                 lane_types: list | None = None):
        super().__init__()
        self.patch_rows  = patch_rows
        self.in_channels = in_channels
        self.use_lane_type_embed = use_lane_type_embed
        if use_lane_type_embed:
            lt = lane_types if lane_types is not None else [0] + [1] * 14 + [2]
            n_types = max(lt) + 1
            self.lane_type_embed = nn.Embedding(n_types, patch_rows)
            self.register_buffer('lane_types', torch.tensor(lt, dtype=torch.long))
        self.proj = nn.Linear(patch_rows * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, window_rows, in_channels)
        B, W, C = x.shape
        num_patches = W // self.patch_rows
        x = x.reshape(B, num_patches, self.patch_rows, C)
        if self.use_lane_type_embed:
            lt = self.lane_type_embed(self.lane_types).T  # (patch_rows, in_channels)
            x = x + lt
        x = x.reshape(B, num_patches, self.patch_rows * C)
        return self.proj(x)  # (B, num_patches, embed_dim)


# ── MAEEncoder ────────────────────────────────────────────────────────────────

class MAEEncoder(nn.Module):
    """
    Transformer encoder used for both MAE pretraining and fine-tuning.

    use_rope=True: RoPE attention (no absolute PE), BPM injected per-patch.
    use_rope=False (default): absolute learned PE, BPM added to CLS token.
    """

    def __init__(self, embed_dim: int = 128, depth: int = 6, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 in_channels: int = IN_CHANNELS, num_patches: int = NUM_PATCHES,
                 patch_rows: int = PATCH_ROWS,
                 use_lane_type_embed: bool = False, use_bpm_cond: bool = False,
                 use_rope: bool = False,
                 lane_types: list | None = None):
        super().__init__()
        self.embed_dim    = embed_dim
        self.use_bpm_cond = use_bpm_cond
        self.use_rope     = use_rope
        self.patch_embed  = PatchEmbed(patch_rows=patch_rows, embed_dim=embed_dim,
                                       in_channels=in_channels,
                                       use_lane_type_embed=use_lane_type_embed,
                                       lane_types=lane_types)
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if not use_rope:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.density_proj = nn.Linear(1, embed_dim)
        if use_bpm_cond:
            self.bpm_proj = nn.Linear(1, embed_dim)

        if use_rope:
            self.transformer = RoPETransformerEncoder(
                embed_dim, num_heads, depth, mlp_ratio=mlp_ratio, dropout=dropout)
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=num_heads,
                dim_feedforward=int(embed_dim * mlp_ratio),
                dropout=dropout, batch_first=True, norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=depth, enable_nested_tensor=False)

        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        if not self.use_rope:
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.density_proj.weight, std=0.02)
        nn.init.zeros_(self.density_proj.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, keep_ids: torch.Tensor | None = None,
                density: torch.Tensor | None = None,
                bpm: torch.Tensor | None = None) -> torch.Tensor:
        B = x.shape[0]
        tokens = self.patch_embed(x)       # (B, num_patches, embed_dim)

        if not self.use_rope:
            tokens = tokens + self.pos_embed

        if keep_ids is not None:
            tokens = tokens[torch.arange(B, device=x.device).unsqueeze(1), keep_ids]

        # BPM: per-patch when use_rope, otherwise onto CLS below
        if bpm is not None and self.use_bpm_cond and self.use_rope:
            b_embed = self.bpm_proj(bpm.unsqueeze(1))   # (B, embed_dim)
            tokens  = tokens + b_embed.unsqueeze(1)     # broadcast to all patches

        cls = self.cls_token.expand(B, -1, -1).clone()
        if density is not None:
            cls = cls + self.density_proj(density.unsqueeze(1)).unsqueeze(1)
        if bpm is not None and self.use_bpm_cond and not self.use_rope:
            cls = cls + self.bpm_proj(bpm.unsqueeze(1)).unsqueeze(1)

        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.transformer(tokens)
        return self.norm(tokens)

    def embed_segment(self, x: torch.Tensor,
                      density: torch.Tensor | None = None,
                      bpm: torch.Tensor | None = None) -> torch.Tensor:
        """Convenience: encode a full window and return the CLS embedding."""
        return self.forward(x, density=density, bpm=bpm)[:, 0]

    def embed_patches(self, x: torch.Tensor,
                      density: torch.Tensor | None = None,
                      bpm: torch.Tensor | None = None) -> torch.Tensor:
        """Return all patch token embeddings (B, num_patches, embed_dim), excluding CLS."""
        return self.forward(x, density=density, bpm=bpm)[:, 1:]


# ── MAEDecoder ────────────────────────────────────────────────────────────────

class MAEDecoder(nn.Module):
    """Lightweight decoder: reconstructs masked patches from encoded context."""

    def __init__(self, encoder_dim: int = 128, decoder_dim: int = 64,
                 num_patches: int = NUM_PATCHES, patch_rows: int = PATCH_ROWS,
                 in_channels: int = IN_CHANNELS, depth: int = 2, num_heads: int = 2):
        super().__init__()
        self.num_patches  = num_patches
        self.patch_size   = patch_rows * in_channels
        self.encoder_proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token   = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed    = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim, nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, self.patch_size)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, encoded: torch.Tensor, keep_ids: torch.Tensor,
                mask_ids: torch.Tensor) -> torch.Tensor:
        B = encoded.shape[0]
        enc_patches = self.encoder_proj(encoded[:, 1:])  # (B, num_keep, decoder_dim)

        tokens = self.mask_token.expand(B, self.num_patches, -1).clone().to(enc_patches.dtype)
        tokens[torch.arange(B, device=encoded.device).unsqueeze(1), keep_ids] = enc_patches
        tokens = tokens + self.pos_embed

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens)  # (B, num_patches, patch_size)


# ── MAEModel ──────────────────────────────────────────────────────────────────

class MAEModel(nn.Module):
    """Combined encoder + decoder for MAE pretraining."""

    def __init__(self, mask_ratio: float = 0.75, encoder_dim: int = 128,
                 decoder_dim: int = 64, encoder_depth: int = 6,
                 decoder_depth: int = 2, num_heads: int = 4,
                 in_channels: int = PRETRAIN_IN_CHANNELS,
                 num_patches: int = NUM_PATCHES,
                 patch_rows: int = PATCH_ROWS,
                 nonzero_weight: float = 4.0,
                 hold_weight: float = 2.0,
                 event_weight: float = 1.0,
                 empty_weight: float = 1.0,
                 density_mask: bool = False,
                 use_lane_type_embed: bool = False,
                 use_bpm_cond: bool = False,
                 use_rope: bool = False,
                 lane_types: list | None = None):
        super().__init__()
        self.mask_ratio     = mask_ratio
        self.in_channels    = in_channels
        self.num_patches    = num_patches
        self.patch_rows     = patch_rows
        self.nonzero_weight = nonzero_weight
        self.hold_weight    = hold_weight
        self.event_weight   = event_weight
        self.empty_weight   = empty_weight
        self.density_mask   = density_mask
        self.use_rope       = use_rope
        self.num_keep = max(1, int(num_patches * (1 - mask_ratio)))

        self.encoder = MAEEncoder(embed_dim=encoder_dim, depth=encoder_depth,
                                  num_heads=num_heads, in_channels=in_channels,
                                  num_patches=num_patches, patch_rows=patch_rows,
                                  use_lane_type_embed=use_lane_type_embed,
                                  use_bpm_cond=use_bpm_cond,
                                  use_rope=use_rope,
                                  lane_types=lane_types)
        self.decoder = MAEDecoder(encoder_dim=encoder_dim, decoder_dim=decoder_dim,
                                  depth=decoder_depth, in_channels=in_channels,
                                  num_patches=num_patches, patch_rows=patch_rows)

    def _random_mask(self, B: int, device,
                     x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x is not None and self.density_mask:
            x_patches = x.reshape(B, self.num_patches, self.patch_rows, self.in_channels)
            counts = (x_patches >= 0.9).sum(dim=(-1, -2)).float()
            noise = torch.rand(B, self.num_patches, device=device) * (counts.sqrt() + 1.0)
        else:
            noise = torch.rand(B, self.num_patches, device=device)
        ids_sort = noise.argsort(dim=1)
        keep_ids = ids_sort[:, :self.num_keep]
        mask_ids = ids_sort[:, self.num_keep:]
        if self.use_rope:
            # Sort so patches enter encoder in original position order;
            # RoPE positions then correspond to sequential chart positions.
            keep_ids = keep_ids.sort(dim=1).values
        return keep_ids, mask_ids

    def forward(
        self,
        x: torch.Tensor,
        density: torch.Tensor | None = None,
        bpm: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float]:
        """Returns (loss, recon_loss_value)."""
        B, W, C = x.shape
        keep_ids, mask_ids = self._random_mask(B, x.device, x)

        encoded = self.encoder(x, keep_ids, density, bpm)
        pred    = self.decoder(encoded, keep_ids, mask_ids)

        target = x.reshape(B, self.num_patches, self.patch_rows * self.in_channels)

        num_masked    = mask_ids.shape[1]
        batch_idx     = torch.arange(B, device=x.device).unsqueeze(1)
        pred_masked   = pred[batch_idx, mask_ids]
        target_masked = target[batch_idx, mask_ids]

        if self.event_weight != self.empty_weight:
            target_rows = target_masked.reshape(B, num_masked, self.patch_rows, self.in_channels)
            pred_rows   = pred_masked.reshape(B, num_masked, self.patch_rows, self.in_channels)
            event_rows  = (target_rows >= 0.9).any(dim=-1, keepdim=True).float()
            weight = self.empty_weight + (self.event_weight - self.empty_weight) * event_rows
            recon_loss = ((pred_rows - target_rows) ** 2 * weight).mean()
        elif self.nonzero_weight > 0:
            mse       = (pred_masked - target_masked) ** 2
            tap_mask  = (target_masked >= 0.9).float()
            hold_mask = ((target_masked > 0.1) & (target_masked < 0.9)).float()
            weight = 1.0 + self.nonzero_weight * tap_mask + self.hold_weight * hold_mask
            recon_loss = (mse * weight).mean()
        else:
            recon_loss = ((pred_masked - target_masked) ** 2).mean()

        return recon_loss, float(recon_loss.detach())
