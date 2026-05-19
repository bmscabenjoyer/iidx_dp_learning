import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import WINDOW_ROWS, NUM_PATCHES, PATCH_ROWS, IN_CHANNELS


class PatchEmbed(nn.Module):
    """Flatten each beat-patch and project to embed_dim."""

    def __init__(self, patch_rows: int = PATCH_ROWS, in_channels: int = IN_CHANNELS,
                 embed_dim: int = 128):
        super().__init__()
        self.patch_rows = patch_rows
        self.proj = nn.Linear(patch_rows * in_channels, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, window_rows, in_channels)
        B = x.shape[0]
        x = x.reshape(B, NUM_PATCHES, self.patch_rows * IN_CHANNELS)
        return self.proj(x)  # (B, NUM_PATCHES, embed_dim)


class MAEEncoder(nn.Module):
    """
    Transformer encoder used for both MAE pretraining and fine-tuning.

    During pretraining: receives only unmasked patches (keep_ids provided).
    During fine-tuning: receives all patches (keep_ids=None).
    The CLS token embedding is the segment representation.
    """

    def __init__(self, embed_dim: int = 128, depth: int = 6, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(embed_dim=embed_dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed  = nn.Parameter(torch.zeros(1, NUM_PATCHES, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # pre-norm (more stable)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, keep_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        x        : (B, WINDOW_ROWS, IN_CHANNELS)
        keep_ids : (B, num_keep) long — indices of unmasked patches; None = keep all
        Returns  : (B, 1 + num_keep, embed_dim)  [cls token first]
        """
        B = x.shape[0]
        tokens = self.patch_embed(x)           # (B, NUM_PATCHES, embed_dim)
        tokens = tokens + self.pos_embed        # add positional info before masking

        if keep_ids is not None:
            # keep only unmasked patches
            tokens = tokens[torch.arange(B, device=x.device).unsqueeze(1), keep_ids]

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = self.transformer(tokens)
        return self.norm(tokens)               # (B, 1+num_keep, embed_dim)

    def embed_segment(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience: encode a full window and return the CLS embedding."""
        return self.forward(x)[:, 0]          # (B, embed_dim)


class MAEDecoder(nn.Module):
    """Lightweight decoder: reconstructs masked patches from encoded context."""

    def __init__(self, encoder_dim: int = 128, decoder_dim: int = 64,
                 num_patches: int = NUM_PATCHES, patch_rows: int = PATCH_ROWS,
                 in_channels: int = IN_CHANNELS, depth: int = 2, num_heads: int = 2):
        super().__init__()
        self.num_patches = num_patches
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
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(decoder_dim)
        self.head = nn.Linear(decoder_dim, patch_rows * in_channels)

        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, encoded: torch.Tensor, keep_ids: torch.Tensor,
                mask_ids: torch.Tensor) -> torch.Tensor:
        """
        encoded  : (B, 1+num_keep, encoder_dim)  — [cls, unmasked patches]
        keep_ids : (B, num_keep)
        mask_ids : (B, num_masked)
        Returns  : (B, num_patches, patch_rows * in_channels)
        """
        B = encoded.shape[0]
        enc_patches = self.encoder_proj(encoded[:, 1:])  # (B, num_keep, decoder_dim)

        # build full-length sequence, inserting encoded patches at kept positions
        tokens = self.mask_token.expand(B, self.num_patches, -1).clone()
        tokens[torch.arange(B, device=encoded.device).unsqueeze(1), keep_ids] = enc_patches
        tokens = tokens + self.pos_embed

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)
        return self.head(tokens)  # (B, num_patches, patch_rows * in_channels)


class MAEModel(nn.Module):
    """Combined encoder + decoder for MAE pretraining."""

    def __init__(self, mask_ratio: float = 0.75, encoder_dim: int = 128,
                 decoder_dim: int = 64, encoder_depth: int = 6,
                 decoder_depth: int = 2, num_heads: int = 4):
        super().__init__()
        self.mask_ratio = mask_ratio
        num_keep = max(1, int(NUM_PATCHES * (1 - mask_ratio)))
        self.num_keep = num_keep

        self.encoder = MAEEncoder(embed_dim=encoder_dim, depth=encoder_depth,
                                  num_heads=num_heads)
        self.decoder = MAEDecoder(encoder_dim=encoder_dim, decoder_dim=decoder_dim,
                                  depth=decoder_depth)

    def _random_mask(self, B: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        noise    = torch.rand(B, NUM_PATCHES, device=device)
        ids_sort = noise.argsort(dim=1)
        keep_ids = ids_sort[:, :self.num_keep]
        mask_ids = ids_sort[:, self.num_keep:]
        return keep_ids, mask_ids

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scalar MSE loss on masked patches."""
        B, W, C = x.shape
        keep_ids, mask_ids = self._random_mask(B, x.device)

        encoded = self.encoder(x, keep_ids)                  # (B, 1+num_keep, D)
        pred    = self.decoder(encoded, keep_ids, mask_ids)  # (B, num_patches, patch*C)

        # target: original patches
        target = x.reshape(B, NUM_PATCHES, PATCH_ROWS * IN_CHANNELS)

        pred_masked   = pred[torch.arange(B, device=x.device).unsqueeze(1), mask_ids]
        target_masked = target[torch.arange(B, device=x.device).unsqueeze(1), mask_ids]
        return F.mse_loss(pred_masked, target_masked)
