# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# plane head implementation for plana3r
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
from plana3r.utils.misc import normalize_quaternion

class CameraHeadSimple(nn.Module):
    """ 
    camera head for plana3r
    Each token outputs: -
    """

    def __init__(self, dec_dim, hidden_dim):
        super().__init__()
        self.mlp_pose_rot = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.mlp_pose_trans = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # self.mlp_fov = nn.Sequential(
        #     nn.Linear(dec_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        # )
        self.fc_trans = nn.Linear(hidden_dim, 3)
        self.fc_rot = nn.Linear(hidden_dim, 4)
        # self.fc_fov = nn.Linear(hidden_dim, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, dec_feat):
        pred_rel_rot_q = self.mlp_pose_rot(dec_feat).permute(0, 2, 1)
        pred_rel_rot_q = self.pool(pred_rel_rot_q).squeeze(-1)
        pred_rel_rot_q = self.fc_rot(pred_rel_rot_q)
        pred_rel_rot_q_normed = normalize_quaternion(pred_rel_rot_q)
        
        pred_rel_trans = self.mlp_pose_trans(dec_feat).permute(0, 2, 1)
        pred_rel_trans = self.pool(pred_rel_trans).squeeze(-1)
        pred_rel_trans = self.fc_trans(pred_rel_trans)

        # pred_fov = self.mlp_fov(dec_feat).permute(0, 2, 1)
        # pred_fov = self.pool(pred_fov).squeeze(-1)
        # pred_fov = self.fc_fov(pred_fov)

        return pred_rel_rot_q_normed, pred_rel_rot_q, pred_rel_trans

class CameraHeadSimple_FoV(nn.Module):
    """ 
    camera head for plana3r
    Each token outputs: -
    """

    def __init__(self, dec_dim, hidden_dim):
        super().__init__()
        self.mlp_fov = nn.Sequential(
            nn.Linear(dec_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_fov = nn.Linear(hidden_dim, 2)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, dec_feat):
        pred_fov = self.mlp_fov(dec_feat).permute(0, 2, 1)
        pred_fov = self.pool(pred_fov).squeeze(-1)
        pred_fov = self.fc_fov(pred_fov)

        return pred_fov