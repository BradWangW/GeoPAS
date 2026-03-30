import torch
import torch.nn as nn
import torch.nn.functional as F

class ContourCNNSelector(nn.Module):
    def __init__(self, num_algorithms, *, dual_head: bool = True, conv_channels=[32, 64, 128]):
        super().__init__()

        self.dual_head = bool(dual_head)

        self.encoder = PlotEncoder(conv_channels=conv_channels)
        self.stats_embed = nn.Linear(3, 16)

        self.plot_attn = PlotAttention(conv_channels[-1] + 16)

        self.dim_embed = nn.Linear(1, 1)

        self.pre_head_dropout = nn.Dropout(p=0.2)

        self.head = nn.Sequential(
            nn.Linear(conv_channels[-1] + 16 + 1, 128),
            nn.ReLU(),
            nn.Linear(128, num_algorithms)
        )

        self.cat_head = None
        if self.dual_head:
            self.cat_head = nn.Sequential(
                nn.Linear(conv_channels[-1] + 16 + 1, 128),
                nn.ReLU(),
                nn.Linear(128, num_algorithms)
            )

    def forward(self, plots, masks, stats, dim):
        """
        plots:  (B, K, r, r)
        masks:  (B, K, r, r)   {0,1}
        stats: (B, K, 3) where stats[...,0] is ell/scale, stats[...,1] is range, stats[...,2] is iqr
        dim:    (B,)  dimension of the problem
        """
        B, K, _, _ = plots.shape

        plots = plots.view(B * K, 1, plots.size(-2), plots.size(-1))
        masks = masks.view(B * K, 1, masks.size(-2), masks.size(-1))

        z = self.encoder(plots, masks)           # (B*K, conv_channels[-1])
        z = z.view(B, K, -1)

        log_stats = torch.log(stats.clamp_min(1e-6))
        s = self.stats_embed(log_stats)

        z = torch.cat([z, s], dim=-1)             # (B,K, conv_channels[-1]+16)

        Z = self.plot_attn(z)                      # (B, conv_channels[-1]+16)

        log_dim = torch.log(dim.unsqueeze(-1).clamp_min(1e-6).to(Z.dtype))  # (B,1)
        d = self.dim_embed(log_dim)                     # (B,1)

        Z = torch.cat([Z, d], dim=-1)  # (B, conv_channels[-1]+16+1)

        Z = self.pre_head_dropout(Z)

        out = self.head(Z)  # (B,num_algorithms)
        cat_logits = self.cat_head(Z) if self.cat_head is not None else None
        return out, cat_logits

class PlotAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, z):
        """
        z: (B, K, D)
        """
        logits = self.score(z).squeeze(-1)     # (B,K)
        weights = torch.softmax(logits, dim=1)
        return (z * weights.unsqueeze(-1)).sum(dim=1)

class PlotEncoder(nn.Module):
    def __init__(self, conv_channels=[32, 64, 128]):
        super().__init__()
        self.block1 = ConvBlock(1, conv_channels[0])
        self.block2 = ConvBlock(conv_channels[0], conv_channels[1])
        self.block3 = ConvBlock(conv_channels[1], conv_channels[2])

        self.spatial_pool = MaskedSpatialAttention(conv_channels[2])

    def forward(self, x, mask):
        """
        x:    (B, 1, r, r)
        mask: (B, 1, r, r)
        """
        x = self.block1(x)
        x, mask = masked_downsample(x, mask)

        x = self.block2(x)
        x, mask = masked_downsample(x, mask)

        x = self.block3(x)

        z = self.spatial_pool(x, mask)   # (B, conv_channels[2])
        return z
    
class MaskedSpatialAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.attn = nn.Conv2d(channels, 1, kernel_size=1)

    def forward(self, x, mask):
        """
        x:    (B, C, H, W)
        mask: (B, 1, H, W), values in {0,1}
        """
        B, C, H, W = x.shape

        logits = self.attn(x)                  # (B,1,H,W)
        logits = logits.flatten(-2)            # (B,1,HW)
        mask_f = mask.flatten(-2)              # (B,1,HW)

        # Mask invalid positions
        logits = logits.masked_fill(mask_f == 0, -1e9)

        # Softmax over valid pixels only
        weights = torch.softmax(logits, dim=-1)
        weights = weights * mask_f

        # Renormalise to avoid NaNs when few valid pixels exist
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-6)

        weights = weights.view(B, 1, H, W)

        pooled = (x * weights).sum(dim=(-2, -1))  # (B,C)
        return pooled

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

def masked_downsample(x, mask, kernel_size=2, stride=2):
    x    = F.max_pool2d(x,    kernel_size, stride)
    mask = F.max_pool2d(mask, kernel_size, stride)
    return x, mask
