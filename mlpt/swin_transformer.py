"""
1D-adapted Swin Transformer for peptide sequence processing.

Implements:
- Patch extraction and linear embedding
- Window-based Multi-head Self-Attention (W-MSA)
- Shifted Window Multi-head Self-Attention (SW-MSA)
- Swin Transformer Block with residual connections and MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding1D(nn.Module):
    """
    Split 1D feature maps into non-overlapping patches and embed them.

    Input:  (batch, channels, seq_len) e.g. (B, 64, 64)
    Output: (batch, num_patches, embed_dim) e.g. (B, 16, 512)
    """

    def __init__(self, in_channels=64, patch_size=4, embed_dim=512):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """x: (B, C, L) -> (B, L/patch_size, embed_dim)"""
        B, C, L = x.shape
        p = self.patch_size
        num_patches = L // p

        # Reshape: (B, C, L) -> (B, num_patches, C * patch_size)
        x = x[:, :, :num_patches * p]  # trim to exact multiple
        x = x.reshape(B, C, num_patches, p)
        x = x.permute(0, 2, 1, 3)  # (B, num_patches, C, p)
        x = x.reshape(B, num_patches, C * p)

        # Linear projection + LayerNorm
        x = self.proj(x)
        x = self.norm(x)
        return x


class WindowAttention1D(nn.Module):
    """
    Window-based multi-head self-attention for 1D sequences.
    Includes learnable relative position bias.
    """

    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(0.1)
        self.proj_drop = nn.Dropout(0.1)

        # Relative position bias table
        # For 1D: positions range from -(window_size-1) to +(window_size-1)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(2 * window_size - 1, num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Compute relative position index for each pair within window
        coords = torch.arange(window_size)
        relative_coords = coords.unsqueeze(0) - coords.unsqueeze(1)  # (W, W)
        relative_coords += window_size - 1  # shift to start from 0
        self.register_buffer("relative_position_index", relative_coords)

    def forward(self, x, mask=None):
        """
        Args:
            x: (num_windows * B, window_size, embed_dim)
            mask: optional attention mask
        Returns:
            (num_windows * B, window_size, embed_dim)
        """
        BW, W, C = x.shape
        qkv = self.qkv(x).reshape(BW, W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each: (BW, num_heads, W, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size, self.window_size, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # (nH, W, W)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(BW // nW, nW, self.num_heads, W, W)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, W, W)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(BW, W, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


def window_partition_1d(x, window_size):
    """
    Partition sequence of patches into non-overlapping windows.

    Args:
        x: (B, L, C)
        window_size: int
    Returns:
        windows: (B * num_windows, window_size, C)
    """
    B, L, C = x.shape
    num_windows = L // window_size
    x = x.view(B, num_windows, window_size, C)
    windows = x.reshape(B * num_windows, window_size, C)
    return windows


def window_reverse_1d(windows, window_size, L):
    """
    Reverse window partition.

    Args:
        windows: (B * num_windows, window_size, C)
        window_size: int
        L: original sequence length
    Returns:
        x: (B, L, C)
    """
    num_windows = L // window_size
    B = windows.shape[0] // num_windows
    x = windows.view(B, num_windows, window_size, -1)
    x = x.reshape(B, L, -1)
    return x


class SwinBlock1D(nn.Module):
    """
    Swin Transformer Block for 1D sequences.

    Consists of:
    - LayerNorm + Window/Shifted-Window MSA + residual
    - LayerNorm + MLP (Linear -> GELU -> Dropout -> Linear -> Dropout) + residual
    """

    def __init__(self, embed_dim, num_heads, window_size, shift_size=0,
                 mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowAttention1D(embed_dim, num_heads, window_size)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        """
        Args:
            x: (B, L, C) where L is number of patches
        Returns:
            (B, L, C)
        """
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x)

        # Cyclic shift for SW-MSA
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=-self.shift_size, dims=1)
            # Compute attention mask for shifted windows
            attn_mask = self._compute_mask(L, x.device)
        else:
            shifted_x = x
            attn_mask = None

        # Partition into windows
        windows = window_partition_1d(shifted_x, self.window_size)

        # Window attention
        attn_windows = self.attn(windows, mask=attn_mask)

        # Reverse partition
        shifted_x = window_reverse_1d(attn_windows, self.window_size, L)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=self.shift_size, dims=1)
        else:
            x = shifted_x

        # Residual + MLP
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x

    def _compute_mask(self, L, device):
        """Compute attention mask for shifted window MSA."""
        # Assign each position to a window region
        num_windows = L // self.window_size
        region_ids = torch.zeros(L, dtype=torch.long, device=device)
        cnt = 0
        # After shift, the boundary positions from different original windows
        # end up in the same shifted window - mask prevents them from attending
        pos = 0
        sizes = []
        if self.shift_size > 0:
            sizes = [L - self.window_size, self.window_size - self.shift_size, self.shift_size]
        else:
            sizes = [L]

        for s in sizes:
            region_ids[pos:pos + s] = cnt
            cnt += 1
            pos += s

        # Partition region_ids into windows
        region_ids = region_ids.view(num_windows, self.window_size)

        # Mask: within each window, positions from different regions get -100
        attn_mask = region_ids.unsqueeze(1) - region_ids.unsqueeze(2)  # (nW, W, W)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
        return attn_mask
