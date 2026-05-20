import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import (WINDOW_ROWS, NUM_PATCHES, PATCH_ROWS, IN_CHANNELS,
                     PRETRAIN_IN_CHANNELS, PRETRAIN_IN_CHANNELS_V3,
                     NUM_PATCHES_V4, PATCH_ROWS_V4)


class PatchEmbed(nn.Module):
    """Flatten each patch and project to embed_dim."""

    def __init__(self, patch_rows: int = PATCH_ROWS, in_channels: int = IN_CHANNELS,
                 embed_dim: int = 128):
        super().__init__()
        self.patch_rows  = patch_rows
        self.in_channels = in_channels
        self.proj = nn.Linear(patch_rows * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, window_rows, in_channels)
        B, W, C = x.shape
        num_patches = W // self.patch_rows
        x = x.reshape(B, num_patches, self.patch_rows * self.in_channels)
        return self.proj(x)  # (B, num_patches, embed_dim)


class MAEEncoder(nn.Module):
    """
    Transformer encoder used for both MAE pretraining and fine-tuning.

    During pretraining: receives only unmasked patches (keep_ids provided).
    During fine-tuning: receives all patches (keep_ids=None).
    The CLS token embedding is the segment representation.
    """

    def __init__(self, embed_dim: int = 128, depth: int = 6, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1,
                 in_channels: int = IN_CHANNELS, num_patches: int = NUM_PATCHES,
                 patch_rows: int = PATCH_ROWS):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed  = PatchEmbed(patch_rows=patch_rows, embed_dim=embed_dim,
                                       in_channels=in_channels)
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed    = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.density_proj = nn.Linear(1, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.density_proj.weight, std=0.02)
        nn.init.zeros_(self.density_proj.bias)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, keep_ids: torch.Tensor | None = None,
                density: torch.Tensor | None = None) -> torch.Tensor:
        """
        x        : (B, WINDOW_ROWS, in_channels)
        keep_ids : (B, num_keep) long — indices of unmasked patches; None = keep all
        density  : (B,) float — log(1 + notes/sec) for this window; None = no conditioning
        Returns  : (B, 1 + num_keep, embed_dim)  [cls token first]
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)       # (B, num_patches, embed_dim)
        tokens = tokens + self.pos_embed

        if keep_ids is not None:
            tokens = tokens[torch.arange(B, device=x.device).unsqueeze(1), keep_ids]

        cls = self.cls_token.expand(B, -1, -1).clone()
        if density is not None:
            d_embed = self.density_proj(density.unsqueeze(1))  # (B, embed_dim)
            cls = cls + d_embed.unsqueeze(1)                   # (B, 1, embed_dim)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.transformer(tokens)
        return self.norm(tokens)           # (B, 1+num_keep, embed_dim)

    def embed_segment(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: encode a full window and return the CLS embedding."""
        return self.forward(x)[:, 0]      # (B, embed_dim)


class MAEDecoder(nn.Module):
    """Lightweight decoder: reconstructs masked patches from encoded context."""

    def __init__(self, encoder_dim: int = 128, decoder_dim: int = 64,
                 num_patches: int = NUM_PATCHES, patch_rows: int = PATCH_ROWS,
                 in_channels: int = IN_CHANNELS, depth: int = 2, num_heads: int = 2):
        super().__init__()
        self.num_patches = num_patches
        self.patch_size  = patch_rows * in_channels
        self.encoder_proj = nn.Linear(encoder_dim, decoder_dim)
        self.mask_token   = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.pos_embed    = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=num_heads,
            dim_feedforward=decoder_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth,
                                                  enable_nested_tensor=False)
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, self.patch_size)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, encoded: torch.Tensor, keep_ids: torch.Tensor,
                mask_ids: torch.Tensor) -> torch.Tensor:
        """
        encoded  : (B, 1+num_keep, encoder_dim)
        keep_ids : (B, num_keep)
        mask_ids : (B, num_masked)
        Returns  : (B, num_patches, patch_size)
        """
        B = encoded.shape[0]
        enc_patches = self.encoder_proj(encoded[:, 1:])  # (B, num_keep, decoder_dim)

        tokens = self.mask_token.expand(B, self.num_patches, -1).clone().to(enc_patches.dtype)
        tokens[torch.arange(B, device=encoded.device).unsqueeze(1), keep_ids] = enc_patches
        tokens = tokens + self.pos_embed

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens)  # (B, num_patches, patch_size)


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
                 zasa_weight: float = 0.1):
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
        self.zasa_weight    = zasa_weight
        self.num_keep = max(1, int(num_patches * (1 - mask_ratio)))

        self.encoder = MAEEncoder(embed_dim=encoder_dim, depth=encoder_depth,
                                  num_heads=num_heads, in_channels=in_channels,
                                  num_patches=num_patches, patch_rows=patch_rows)
        self.decoder = MAEDecoder(encoder_dim=encoder_dim, decoder_dim=decoder_dim,
                                  depth=decoder_depth, in_channels=in_channels,
                                  num_patches=num_patches, patch_rows=patch_rows)

        if zasa_weight > 0:
            self.zasa_head = nn.Sequential(
                nn.Linear(encoder_dim, 32),
                nn.GELU(),
                nn.Linear(32, 1),
            )
        else:
            self.zasa_head = None

    def _random_mask(self, B: int, device,
                     x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if x is not None and self.density_mask:
            # Denser patches get higher noise → sorted later → more likely masked.
            x_patches = x.reshape(B, self.num_patches, self.patch_rows, self.in_channels)
            counts = (x_patches >= 0.9).sum(dim=(-1, -2)).float()  # (B, num_patches)
            noise = torch.rand(B, self.num_patches, device=device) * (counts.sqrt() + 1.0)
        else:
            noise = torch.rand(B, self.num_patches, device=device)
        ids_sort = noise.argsort(dim=1)
        keep_ids = ids_sort[:, :self.num_keep]
        mask_ids = ids_sort[:, self.num_keep:]
        return keep_ids, mask_ids

    def forward(
        self,
        x: torch.Tensor,
        density: torch.Tensor | None = None,
        zasa_label: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, float, float]:
        """Returns (total_loss, recon_loss_value, zasa_loss_value)."""
        B, W, C = x.shape
        keep_ids, mask_ids = self._random_mask(B, x.device, x)

        encoded = self.encoder(x, keep_ids, density)
        pred    = self.decoder(encoded, keep_ids, mask_ids)

        target = x.reshape(B, self.num_patches, self.patch_rows * self.in_channels)

        num_masked    = mask_ids.shape[1]
        batch_idx     = torch.arange(B, device=x.device).unsqueeze(1)
        pred_masked   = pred[batch_idx, mask_ids]    # (B, num_masked, patch_size)
        target_masked = target[batch_idx, mask_ids]

        if self.event_weight != self.empty_weight:
            # Row-level event weighting (v4): event rows cost more than empty rows.
            target_rows = target_masked.reshape(B, num_masked, self.patch_rows, self.in_channels)
            pred_rows   = pred_masked.reshape(B, num_masked, self.patch_rows, self.in_channels)
            event_rows  = (target_rows >= 0.9).any(dim=-1, keepdim=True).float()
            weight = self.empty_weight + (self.event_weight - self.empty_weight) * event_rows
            recon_loss = ((pred_rows - target_rows) ** 2 * weight).mean()
        elif self.nonzero_weight > 0:
            # Cell-level nonzero weighting (v2/v3).
            mse       = (pred_masked - target_masked) ** 2
            tap_mask  = (target_masked >= 0.9).float()
            hold_mask = ((target_masked > 0.1) & (target_masked < 0.9)).float()
            weight = 1.0 + self.nonzero_weight * tap_mask + self.hold_weight * hold_mask
            recon_loss = (mse * weight).mean()
        else:
            recon_loss = ((pred_masked - target_masked) ** 2).mean()

        total_loss    = recon_loss
        zasa_loss_val = 0.0

        if self.zasa_head is not None and zasa_label is not None:
            cls_emb   = encoded[:, 0]
            zasa_pred = self.zasa_head(cls_emb).squeeze(-1)
            valid     = ~torch.isnan(zasa_label)
            if valid.any():
                zasa_loss     = F.mse_loss(zasa_pred[valid], zasa_label[valid])
                total_loss    = total_loss + self.zasa_weight * zasa_loss
                zasa_loss_val = float(zasa_loss.detach())

        return total_loss, float(recon_loss.detach()), zasa_loss_val
