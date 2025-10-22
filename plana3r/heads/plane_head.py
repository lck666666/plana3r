# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# plane head implementation for plana3r
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

class PlaneHead(nn.Module):
    """ 
    plane head for plana3r
    Each token outputs: -
    """

    def __init__(self, dec_dim, hidden_dim):
        super().__init__()
        self.mlp_center_depth = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.ReLU()
        )
        self.mlp_radii = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.ReLU()
        )
        self.mlp_rot = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.Linear(hidden_dim, 4)
        )

        for mlp in [self.mlp_center_depth, self.mlp_radii]:
            nn.init.kaiming_normal_(mlp[0].weight, nonlinearity='relu')
            nn.init.zeros_(mlp[0].bias)
            nn.init.xavier_uniform_(mlp[2].weight)
            nn.init.constant_(mlp[2].bias, 0.1)
            
        nn.init.xavier_uniform_(self.mlp_rot[0].weight)
        nn.init.xavier_uniform_(self.mlp_rot[1].weight)

    def forward(self, dec_feat, view_tag):
        center = self.mlp_center_depth(dec_feat)   #torch.Size([B, n*n, 1])
        radii = self.mlp_radii(dec_feat)  #torch.Size([B, n*n, 2])
        rot = self.mlp_rot(dec_feat)   #torch.Size([B, n*n, 4])

        return {
            f'pred_center_{view_tag}': center,
            f'pred_center_depth_{view_tag}': center,
            f'pred_radii_{view_tag}': radii,
            f'pred_rot_{view_tag}': rot,
        }
