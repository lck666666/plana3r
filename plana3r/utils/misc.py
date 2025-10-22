# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilitary functions for DUSt3R
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('planar_splatting')
from planar_splatting.utils import graphics_utils
import numpy as np

def _interleave_imgs(img1, img2):
    res = {}
    for key, value1 in img1.items():
        value2 = img2[key]
        if isinstance(value1, torch.Tensor):
            value = torch.stack((value1, value2), dim=1).flatten(0, 1)
        else:
            value = [x for pair in zip(value1, value2) for x in pair]
        res[key] = value
    return res

def make_batch_symmetric(batch):
    view1, view2 = batch
    view1, view2 = (_interleave_imgs(view1, view2), _interleave_imgs(view2, view1))
    return view1, view2

def fill_default_args(kwargs, func):
    import inspect  # a bit hacky but it works reliably
    signature = inspect.signature(func)

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue
        kwargs.setdefault(k, v.default)

    return kwargs


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def is_symmetrized(gt1, gt2):
    x = gt1['instance']
    y = gt2['instance']
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def flip(tensor):
    """ flip so that tensor[0::2] <=> tensor[1::2] """
    return torch.stack((tensor[1::2], tensor[0::2]), dim=1).flatten(0, 1)


def interleave(tensor1, tensor2):
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W))
        return res

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return transposed(head(decout, (W, H)))

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W))
        p_result = transposed(head(selout(is_portrait), (W, H)))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no


def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}


def invalid_to_nans(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float('nan')
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr


def invalid_to_zeros(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/util/misc.py
# and from https://github.com/IceTTTb/PlaneTR3D/
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
from typing import List, Optional
import torch.distributed as dist
import torchvision
from torch import Tensor

import numpy as np

def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size # [b, num_queries, 480, 640]
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device) # ep torch.Size([1, 23, 480, 640])
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask): # torch.Size([1, 480, 640])
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_coordinate_map(dataset_name, device, h=192, w=256):
    
    if "scannet" in dataset_name:
        focal_length = 517.97
        offset_x = 320
        offset_y = 240

        K = [[focal_length, 0, offset_x],
            [0, focal_length, offset_y],
            [0, 0, 1]]

    elif "nyuv2" in dataset_name:
        focal_length = 5.8262448167737955e+02
        offset_x = 3.1304475870804731e+02
        offset_y = 2.3844389626620386e+02


        K = [[focal_length, 0, offset_x],
            [0, focal_length, offset_y],
            [0, 0, 1]]

    K_inv = np.linalg.inv(np.array(K))

    K = torch.FloatTensor(K).to(device)
    K_inv = torch.FloatTensor(K_inv).to(device)


    x = torch.arange(w, dtype=torch.float32).view(1, w) / w * 640
    y = torch.arange(h, dtype=torch.float32).view(h, 1) / h * 480

    x = x.to(device)
    y = y.to(device)
    xx = x.repeat(h, 1)
    yy = y.repeat(1, w)
    xy1 = torch.stack((xx, yy, torch.ones((h, w), dtype=torch.float32).to(device)))  # (3, h, w)
    xy1 = xy1.view(3, -1)  # (3, h*w)

    k_inv_dot_xy1 = torch.matmul(K_inv, xy1)  # (3, h*w)
    return k_inv_dot_xy1

def get_raster_cameras(intrinsics, poses, rgbs, mono_depth=None, normal_local=None, image_path=None, idx=None, depth_patches=None,normal_patches=None,depth_patches_low=None,normal_patches_low=None):
    zfar = 10.
    znear = 0.01
    height,width = rgbs.shape[1:3]
    focal_length_x = intrinsics[0,0]
    focal_length_y = intrinsics[1,1]
    FovY = graphics_utils.focal2fov(focal_length_y, height)
    FovX = graphics_utils.focal2fov(focal_length_x, width)

    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    c2w = poses  # 4, 4 
    w2c = c2w.inverse()  # 4, 4
    w2c_right = w2c.T

    world_view_transform = w2c_right.clone().float()
    projection_matrix = graphics_utils.getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1).cuda().float()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]
    uv = np.mgrid[0:height, 0:width].astype(np.int32)
    uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float().cuda()
    uv = uv.reshape(2, -1).transpose(1, 0)  # h*w, 2
    cam_info = {
                "intrinsic": intrinsics.clone(),
                "pose": c2w.clone(),  # camera to world
                "raster_cam_w2c": world_view_transform.clone(),
                "raster_cam_proj": projection_matrix.clone(),
                "raster_cam_fullproj": full_proj_transform.clone(),
                "raster_cam_center": camera_center.clone(),
                "raster_cam_FovX": torch.tensor([FovX]).cuda().clone(),
                "raster_cam_FovY": torch.tensor([FovY]).cuda().clone(),
                "raster_img_center": torch.tensor([cx, cy]).cuda().clone(),
            }
    gt_info = {
        "rgb": rgbs.clone(),
        "image_path": image_path
    }

    if mono_depth is not None:
        gt_info["mono_depth"] = mono_depth.float().cuda().clone()
    if normal_local is not None:
        gt_info["mono_normal_local"] = normal_local.float().cuda().clone()
    if depth_patches is not None:
        gt_info["patch_depth_high"] = depth_patches.float().cuda().clone()
    if normal_patches is not None:
        gt_info["patch_normal_high"] = normal_patches.float().cuda().clone()
    if depth_patches_low is not None:
        gt_info["patch_depth_low"] = depth_patches_low.float().cuda().clone()
    if normal_patches_low is not None:
        gt_info["patch_normal_low"] = normal_patches_low.float().cuda().clone()
    if idx is not None:
        gt_info['index'] = idx

    return ViewInfo(cam_info, gt_info)

def get_raster_cameras_simple(intrinsics, poses, height, width):
    zfar = 10.
    znear = 0.01
    focal_length_x = intrinsics[0,0]
    focal_length_y = intrinsics[1,1]
    FovY = graphics_utils.focal2fov(focal_length_y, height)
    FovX = graphics_utils.focal2fov(focal_length_x, width)

    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    c2w = poses  # 4, 4 
    w2c = c2w.inverse()  # 4, 4
    w2c_right = w2c.T

    world_view_transform = w2c_right.clone().float()
    projection_matrix = graphics_utils.getProjectionMatrix(znear=znear, zfar=zfar, fovX=FovX, fovY=FovY).transpose(0,1).cuda().float()
    full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    camera_center = world_view_transform.inverse()[3, :3]

    cam_info = {
                "intrinsic": intrinsics.clone(),
                "pose": c2w.clone(),  # camera to world
                "raster_cam_w2c": world_view_transform.clone(),
                "raster_cam_proj": projection_matrix.clone(),
                "raster_cam_fullproj": full_proj_transform.clone(),
                "raster_cam_center": camera_center.clone(),
                "raster_cam_FovX": torch.tensor([FovX]).cuda().clone(),
                "raster_cam_FovY": torch.tensor([FovY]).cuda().clone(),
                "raster_img_center": torch.tensor([cx, cy]).cuda().clone(),
            }
    gt_info = {
        "image_path": ''
    }
    return ViewInfo(cam_info, gt_info)

def fov_to_focal(fov, h, w):
    if fov.dim() == 1:
        fov = fov[None]  # b, 2
    elif fov.dim() == 2:
        pass
    else:
        raise NotImplementedError
    fov_h = fov[..., 0]
    fov_w = fov[..., 1]
    fy = (h / 2.0) / torch.tan(fov_h / 2.0)
    fx = (w / 2.0) / torch.tan(fov_w / 2.0)

    return fy, fx

def fov_to_intrinsic_matrix(fov, h, w):
    fy, fx = fov_to_focal(fov, h, w)
    bs = fy.shape[0]
    pred_camera_intrinsics = torch.eye(3).cuda().unsqueeze(0).repeat(bs, 1, 1)  # bs, 3, 3
    pred_camera_intrinsics[:, 0, 0] = fx
    pred_camera_intrinsics[:, 1, 1] = fy
    pred_camera_intrinsics[:, 0, 2] = w / 2.
    pred_camera_intrinsics[:, 1, 2] = h / 2.
    return pred_camera_intrinsics

class ViewInfo(nn.Module):
    def __init__(self, cam_info, gt_info):
        super().__init__()
        self.intrinsic = cam_info['intrinsic'].cuda()
        self.pose = cam_info['pose'].cuda()
        self.raster_cam_w2c = cam_info['raster_cam_w2c'].cuda()
        self.raster_cam_proj = cam_info['raster_cam_proj'].cuda()
        self.raster_cam_fullproj = cam_info['raster_cam_fullproj'].cuda()
        self.raster_cam_center = cam_info['raster_cam_center'].cuda()
        self.raster_cam_FovX = cam_info['raster_cam_FovX'].cpu().item()
        self.raster_cam_FovY = cam_info['raster_cam_FovY'].cpu().item()
        self.tanfovx = math.tan(self.raster_cam_FovX  * 0.5)
        self.tanfovy = math.tan(self.raster_cam_FovY * 0.5)
        self.raster_img_center = cam_info['raster_img_center'].cuda()

        if 'rgb' in gt_info and gt_info['rgb'] is not None:
            self.rgb = gt_info['rgb'].cuda()
        else:
            self.rgb = None
        self.image_path = gt_info['image_path']

        if 'mono_depth' in gt_info:
            self.mono_depth = gt_info['mono_depth'].cuda()
        if 'mono_normal_local' in gt_info:
            self.mono_normal_local = gt_info['mono_normal_local'].cuda()
        if 'index' in gt_info:
            self.index = gt_info['index']
        if 'patch_depth_high' in gt_info:
            self.patch_depth_high = gt_info['patch_depth_high'].cuda()
        if 'patch_normal_high' in gt_info:
            self.patch_normal_high = gt_info['patch_normal_high'].cuda()
        if 'patch_depth_low' in gt_info:
            self.patch_depth_low = gt_info['patch_depth_low'].cuda()
        if 'patch_normal_low' in gt_info:
            self.patch_normal_low = gt_info['patch_normal_low'].cuda()
        
        self.scale = 1.0
        self.shift = 0.0
        self.plane_depth = None

def create_rotation_about_normal(normal, angle_rad):
    """
    创建绕法线旋转的四元数（自动归一化）
    批量创建绕法线旋转的四元数
    输入：
        normal: [n*n, 3] 法线向量
        angle_rad: [n*n, 1] 旋转弧度
    输出：
        quaternions: [n*n, 4] 四元数张量
    """
    # 归一化法线
    norm = torch.norm(normal, dim=1, keepdim=True)
    n_normalized = normal / (norm + 1e-8)  # 防止除零
    
    # 计算四元数分量
    half_angle = angle_rad / 2.0
    cos_half = torch.cos(half_angle)  # [n*n, 1]
    sin_half = torch.sin(half_angle)  # [n*n, 1]
    
    # 构建四元数张量
    quaternions = torch.cat([
        cos_half,                    # w分量
        n_normalized[:, 0:1] * sin_half,  # x分量
        n_normalized[:, 1:2] * sin_half,  # y分量
        n_normalized[:, 2:3] * sin_half   # z分量
    ], dim=1)
    
    return quaternions
    
def normalize_quaternion(q: torch.Tensor) -> torch.Tensor:
    """
    归一化四元数（支持任意批量维度）
    输入形状: 
      - [4]           (单个四元数)
      - [n, 4]        (批量四元数)
      - [batch, ..., 4] (高维批量)
    输出形状: 与输入相同
    """
    # 确定需要归一化的维度（最后一个维度）
    dim = -1
    
    # 计算范数（自动广播处理任意维度）
    norms = torch.linalg.norm(q, ord=2, dim=dim, keepdim=True)  # 结果形状: [..., 1]
    
    # 防止除以0（添加极小值epsilon）
    epsilon = torch.tensor(1e-12, device=q.device, dtype=q.dtype)
    normalized_q = q / torch.maximum(norms, epsilon)
    
    return normalized_q