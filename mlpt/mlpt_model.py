"""
Multi-Level Pooling-based Transformer (MLPT) for Antigenic Peptide Prediction.

Assembles the full model pipeline:
1. ADMAM module for multi-scale feature extraction from physicochemical features
2. Patch embedding to tokenize the 1D feature map
3. Swin Transformer Block 1 (W-MSA) for local attention
4. K-T feature concatenation for antigenicity-aware representation
5. Swin Transformer Block 2 (SW-MSA) for shifted-window attention
6. Multi-level pooling (avg + max + attention) for sequence-level representation
7. Classification head for 6-class prediction
"""

import torch
import torch.nn as nn

from .admam import ADMAM
from .swin_transformer import PatchEmbedding1D, SwinBlock1D
from . import config


class AttentionPooling(nn.Module):
    """Learnable attention-weighted pooling over the sequence dimension."""

    def __init__(self, embed_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, C)
        Returns:
            (B, C)
        """
        weights = self.attention(x)  # (B, L, 1)
        weights = torch.softmax(weights, dim=1)
        return (x * weights).sum(dim=1)


class MLPTModel(nn.Module):
    """
    Multi-Level Pooling-based Transformer.

    Architecture:
        physicochemical_features (B, 39, 64) -> ADMAM -> (B, 64, 64)
        -> PatchEmbedding -> (B, 16, 512)
        -> Swin Block 1 (W-MSA, depth layers) -> (B, 16, 512)
        -> Concatenate K-T features -> project -> (B, 16, 512)
        -> Swin Block 2 (SW-MSA, depth layers) -> (B, 16, 512)
        -> Multi-Level Pooling (avg + max + attn) -> (B, 1536)
        -> Classification Head -> (B, num_classes)
    """

    def __init__(
        self,
        num_classes=config.NUM_CLASSES,
        max_seq_len=config.MAX_SEQ_LEN,
        num_phys_features=config.NUM_PHYSICOCHEMICAL_FEATURES,
        admam_out_channels=config.ADMAM_OUT_CHANNELS,
        patch_size=config.PATCH_SIZE,
        embed_dim=config.EMBEDDING_DIM,
        num_heads=config.SWIN_NUM_HEADS,
        window_size=config.SWIN_WINDOW_SIZE,
        depth=config.SWIN_DEPTH,
        mlp_ratio=config.MLP_RATIO,
        drop_rate=config.DROP_RATE,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = max_seq_len // patch_size  # 64/4 = 16

        # --- Stage 1: ADMAM for multi-scale feature extraction ---
        self.admam = ADMAM(
            in_channels=num_phys_features,
            out_channels=admam_out_channels,
        )

        # --- Stage 2: Patch Embedding ---
        self.patch_embed = PatchEmbedding1D(
            in_channels=admam_out_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # --- Stage 3: Swin Transformer Block 1 (W-MSA) ---
        self.swin_block1 = nn.ModuleList([
            SwinBlock1D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0,  # W-MSA (no shift)
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
            )
            for _ in range(depth)
        ])

        # --- Stage 4: K-T Feature Fusion ---
        # K-T scores: (B, 64) -> reshape to (B, 16, 4) -> project to embed_dim-sized vector
        kt_patch_dim = patch_size  # each patch gets `patch_size` K-T values
        self.kt_proj = nn.Linear(kt_patch_dim, 64)
        # After concatenation: embed_dim + 64 -> project back to embed_dim
        self.kt_fusion = nn.Sequential(
            nn.Linear(embed_dim + 64, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # --- Stage 5: Swin Transformer Block 2 (SW-MSA) ---
        self.swin_block2 = nn.ModuleList([
            SwinBlock1D(
                embed_dim=embed_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=window_size // 2,  # SW-MSA (shifted)
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
            )
            for _ in range(depth)
        ])

        # --- Stage 6: Multi-Level Pooling ---
        self.attn_pool = AttentionPooling(embed_dim)
        # avg_pool + max_pool + attn_pool -> 3 * embed_dim

        # --- Stage 7: Classification Head ---
        pooled_dim = embed_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(pooled_dim, 256),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, encoded_seq, phys_features, kt_scores):
        """
        Args:
            encoded_seq: (B, max_seq_len) int tensor — not used directly by MLPT
                         (kept for API compatibility; embedding-based models could use it)
            phys_features: (B, 39, max_seq_len) float tensor — physicochemical features
            kt_scores: (B, max_seq_len) float tensor — K-T antigenicity scores

        Returns:
            logits: (B, num_classes)
        """
        B = phys_features.shape[0]

        # Stage 1: ADMAM
        # phys_features: (B, 39, 64) -> (B, 64, 64)
        x = self.admam(phys_features)

        # Stage 2: Patch Embedding
        # (B, 64, 64) -> (B, 16, 512)
        x = self.patch_embed(x)

        # Stage 3: Swin Block 1 (W-MSA)
        for block in self.swin_block1:
            x = block(x)

        # Stage 4: K-T Feature Fusion
        # Reshape K-T scores into patches: (B, 64) -> (B, 16, 4)
        kt = kt_scores[:, :self.num_patches * self.patch_embed.patch_size]
        kt = kt.reshape(B, self.num_patches, -1)  # (B, 16, 4)
        kt = self.kt_proj(kt)  # (B, 16, 64)
        # Concatenate and project
        x = torch.cat([x, kt], dim=-1)  # (B, 16, 576)
        x = self.kt_fusion(x)  # (B, 16, 512)

        # Stage 5: Swin Block 2 (SW-MSA)
        for block in self.swin_block2:
            x = block(x)

        # Stage 6: Multi-Level Pooling
        avg_pool = x.mean(dim=1)  # (B, 512)
        max_pool = x.max(dim=1).values  # (B, 512)
        attn_pool = self.attn_pool(x)  # (B, 512)
        pooled = torch.cat([avg_pool, max_pool, attn_pool], dim=-1)  # (B, 1536)

        # Stage 7: Classification
        logits = self.classifier(pooled)
        return logits
