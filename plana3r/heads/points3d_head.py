# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# plane head implementation for plana3r
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from plana3r.heads.postprocess import postprocess

class LinearPts3d (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, net, has_conf=False):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf

        self.proj = nn.Linear(net.dec_embed_dim, (3 + has_conf)*self.patch_size**2)

    def setup(self, croconet):
        pass

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3+x,H,W

        # permute + norm depth
        return postprocess(feat, self.depth_mode, self.conf_mode)

class Pts3dMLP (nn.Module):
    """ 
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, dec_dim, hidden_dim, net, has_conf=False):
        super().__init__()
        self.patch_size = net.patch_embed.patch_size[0]
        self.depth_mode = net.depth_mode
        self.conf_mode = net.conf_mode
        self.has_conf = has_conf

        self.mlp_pts3d = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, (3 + has_conf)*self.patch_size**2),
        )

    def forward(self, dec_feat, view_tag, img_shape):
        H, W = img_shape
        B, S, D = dec_feat.shape

        feat = self.mlp_pts3d(dec_feat)   #torch.Size([B, n*n, 1])

        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3+x,H,W

        # permute + norm depth
        return postprocess(feat, self.depth_mode, self.conf_mode)