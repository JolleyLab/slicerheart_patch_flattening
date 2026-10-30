"""Surrogate network: canonical boundary curve + surface samples -> template outline residual.

Architecture (small on purpose; ~0.4 M parameters, trains on CPU in minutes):
  boundary encoder   circular 1D CNN over the K boundary points (3 -> 64 -> 128 channels)
  surface encoder    PointNet-style shared MLP over M surface points (6 -> 64 -> 128 -> 256), max-pool
  fusion             global vector [maxpool(boundary features), surface code] broadcast to every
                     boundary point and concatenated with its local features
  decoder            circular 1D CNN -> per-point 2D residual (mm)
  scalar head        MLP on the global vector -> [suture mismatch (mm), total |K| (deg), n darts]

Circular convolutions respect the closed boundary; the canonical frame removes rigid
motion, so no rotation augmentation is needed. The output is added to the projected
boundary to give the predicted outline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as Fnn


class CircConv(nn.Module):
    def __init__(self, cin, cout, k=5):
        super().__init__()
        self.pad = k // 2
        self.conv = nn.Conv1d(cin, cout, k)

    def forward(self, x):                       # x: (B, C, K)
        x = Fnn.pad(x, (self.pad, self.pad), mode="circular")
        return self.conv(x)


class OutlineSurrogate(nn.Module):
    def __init__(self, hidden: int = 128, n_scalars: int = 3):
        super().__init__()
        self.b1 = CircConv(3, 64); self.b2 = CircConv(64, hidden); self.b3 = CircConv(hidden, hidden)
        self.s1 = nn.Linear(6, 64); self.s2 = nn.Linear(64, hidden); self.s3 = nn.Linear(hidden, 256)
        g = hidden + 256
        self.d1 = CircConv(hidden + g, hidden); self.d2 = CircConv(hidden, hidden); self.d3 = CircConv(hidden, 2)
        self.scalar_head = nn.Sequential(nn.Linear(g, 128), nn.GELU(), nn.Linear(128, n_scalars))

    def forward(self, boundary, surface):
        # boundary (B,K,3), surface (B,M,6)
        x = boundary.transpose(1, 2)
        h = Fnn.gelu(self.b1(x)); h = Fnn.gelu(self.b2(h)); h = Fnn.gelu(self.b3(h))      # (B,H,K)
        s = Fnn.gelu(self.s1(surface)); s = Fnn.gelu(self.s2(s)); s = self.s3(s)            # (B,M,256)
        s = s.max(dim=1).values                                                            # (B,256)
        g = torch.cat([h.max(dim=2).values, s], dim=1)                                      # (B,H+256)
        gb = g[:, :, None].expand(-1, -1, h.shape[2])
        z = torch.cat([h, gb], dim=1)
        z = Fnn.gelu(self.d1(z)); z = Fnn.gelu(self.d2(z)); resid = self.d3(z).transpose(1, 2)   # (B,K,2)
        return resid, self.scalar_head(g)
