# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# plane head implementation for plana3r
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F

class UpsampleHeadSimple(nn.Module):
    """ 
    upsample head for plana3r
    Each token outputs: -
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv_trans1 = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding
        )

    def forward(self, dec_feat, shape, patch_size):
        assert shape.ndim == 1
        batch_size, seq_len, feat_dim = dec_feat.shape
        h = shape[0].item() // patch_size
        w = shape[1].item() // patch_size
        dec_feat = dec_feat.view(batch_size, h, w, feat_dim)
        dec_feat = dec_feat.permute(0, 3, 1, 2)   # (2, 768, h/16, w/16)
        dec_feat_upsampled = self.conv_trans1(dec_feat)  # (2, 768, 2h/16, 2w/16)
        dec_feat_upsampled = dec_feat_upsampled.permute(0, 2, 3, 1)  # (2, 2h/16, 2w/16, 768)
        dec_feat_upsampled = dec_feat_upsampled.view(batch_size, -1, feat_dim)  # (2, n*n, 768)

        return dec_feat_upsampled
