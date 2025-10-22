import os
import math
from typing import List, Optional, Union, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
import random
import open3d as o3d
import itertools
import time
import glob
import pickle
import quaternion
from planar_splatting.utils.model_util import quaternion_mult, quat_to_rot


def gpu_merge_overlapped_planes(M):
    M = M | M.t()
    adj = M.to(torch.float32)
    n = adj.size(0)
    label = torch.eye(n, dtype=torch.float32, device=M.device)
    while True:
        new_label = label @ adj
        new_label = (new_label > 0).to(torch.float32)
        if torch.allclose(new_label, label):
            break
        label = new_label
    mask = label.bool()
    unique_masks = torch.unique(mask, dim=0, sorted=False)
    groups = []
    for m in unique_masks:
        group = torch.where(m)[0].cpu().tolist()
        groups.append(group)
    return groups

def check_cc_plane(
    plane_center,        # 所有平面的中心点 (S, 3)
    plane_normal,        # 所有平面的法向量 (S, 3)
    plane_pts_dense,      # 所有平面的密集点 (S, k, 3)
    dist_thresh = 0.02,
    delta = 0.02,
    area_thresh = 0.1 * 0.1,
    area_thresh2 = 0.3*0.3
):
    """
    计算平面a的四个角点到平面b的投影点（向量化实现）
    参数:
        plane_pts_dense: 平面的密集点坐标 (S, k, 3)
        plane_normal: 平面的法向量 (S, 3)
        plane_center: 平面的中心点 (S, 3)
    返回:
        plane_ins_assignment: 
    """    
    # 计算所有平面a到所有平面b的投影
    # 扩展法向量和中心点为(S, S, 3)
    n_b = plane_normal.unsqueeze(0)      # (1, S, 3) -> 广播到(S, S, 3)
    p_b = plane_center.unsqueeze(0)       # (1, S, 3) -> 广播到(S, S, 3)
    
    # 计算每个平面b的常数项 d = -n_b·p_b -> (S, S)
    d_b = -torch.sum(n_b * p_b, dim=2)    # (S, S)
    
    # 计算每个目标平面法向量的模长平方 (S, S)
    n_norm_sq = torch.sum(n_b**2, dim=2) + 1e-8  # (S, S)
    
    # 提取所有平面a的四个角点 (S, 4, 3)
    corners = plane_pts_dense[:, :4, :]  # (S, 4, 3)
    # 扩展为 (S, 1, 4, 3) 以便与目标平面b (S, S, ...) 广播
    corners_exp = corners.unsqueeze(1)    # (S, 1, 4, 3)
    
    # 向量化计算投影点:
    # 1. 计算 (q·n_b) 部分 -> (S, S, 4)
    #   n_b: (S, S, 3) -> 扩展为(S, S, 1, 3)
    #   corners_exp: (S, 1, 4, 3) -> 扩展为(S, S, 4, 3)
    #   点积：对最后一维求和 -> (S, S, 4)
    dot_products = torch.sum(corners_exp * n_b.unsqueeze(2), dim=-1)  # (S, S, 4)
    
    # 2. 计算分子: (q·n_b) + d_b -> (S, S, 4)
    numerators = dot_products + d_b.unsqueeze(2)  # (S, S, 4)
    
    # 3. 计算投影: q - [分子/n_norm_sq] * n_b
    #   分子: (S, S, 4, 1) 扩展

    #   n_b: (S, S, 1, 3) 扩展
    #   分母: (S, S, 1, 1) 扩展
    projected_points = corners_exp - (numerators.unsqueeze(-1) * n_b.unsqueeze(2)) / n_norm_sq.unsqueeze(-1).unsqueeze(-1)
    
    # 4. 计算投影点到平面中心的距离
    plane_axis_x = plane_pts_dense[:, 1, :] - plane_pts_dense[:, 0, :] # (S, 3)
    plane_axis_y = plane_pts_dense[:, 2, :] - plane_pts_dense[:, 1, :] # (S, 3)
    plane_radii_x = torch.norm(plane_axis_x, dim=1) / 2. # (S,)
    plane_radii_y = torch.norm(plane_axis_y, dim=1) / 2. # (S,)

    plane_axis_x_normed = F.normalize(plane_axis_x, dim=-1)
    plane_axis_y_normed = F.normalize(plane_axis_y, dim=-1)

    vec_proj_to_center = projected_points - p_b.unsqueeze(2) # (S, S, 4, 3)
    dist_to_center_x = torch.abs(torch.sum(vec_proj_to_center * plane_axis_x_normed.unsqueeze(1).unsqueeze(2), dim=-1)) # (S, S, 4)
    dist_to_center_y = torch.abs(torch.sum(vec_proj_to_center * plane_axis_y_normed.unsqueeze(1).unsqueeze(2), dim=-1)) # (S, S, 4)

    # 5. 计算垂直距离
    corner_proj_dist = torch.norm(projected_points - corners_exp, dim=-1) # (S, S, 4)
    projected_points_mask = corner_proj_dist < dist_thresh # (S, S, 4)

    # calculate cc mask
    cc_mask = (dist_to_center_x < plane_radii_x.unsqueeze(1).unsqueeze(2)+delta) & (dist_to_center_y < plane_radii_y.unsqueeze(1).unsqueeze(2)+delta)
    cc_mask = cc_mask & projected_points_mask
    cc_mask = cc_mask.sum(dim=-1) > 0

    final_cc_mask  = cc_mask | cc_mask.t()    
    final_cc_mask.fill_diagonal_(True)

    merged_groups = gpu_merge_overlapped_planes(final_cc_mask)

    plane_ins_assignment = torch.zeros(plane_center.shape[0], dtype=torch.int).cuda()
    label = 1
    for group in merged_groups:
        group = torch.tensor(group).int().cuda()
        plane_radii_x_group = plane_radii_x[group]
        plane_radii_y_group = plane_radii_y[group]
        
        area = (plane_radii_x_group * plane_radii_y_group * 4).sum()
        if len(group) <= 2:
            area_thresh_true = area_thresh2
        else:
            area_thresh_true = area_thresh
        if area < area_thresh_true:
            continue
        plane_ins_assignment[group] = label
        label += 1
    return plane_ins_assignment

def group_plane_via_normal_vectorized(
    planes_normal: torch.Tensor,
    planes_ins_assignment_masked: Optional[torch.Tensor]=None,
    normal_angle_thresh: float = 25,
    avg_type: str = 'mean',
    precompued_plane_id: Optional[torch.Tensor]=None
) -> torch.Tensor:
    """
    Group planes based on their normal vectors.
    
    Args:
        planes_normal (torch.Tensor): Point normals, shape (S, 3)
        planes_ins_assignment_masked (torch.Tensor): Point instance assignments, shape (S,)
        normal_angle_thresh (float): Angle threshold for normal grouping (in degrees)
        use_mean (bool): Whether to use mean or median for plane normal calculation
        
    Returns:
        torch.Tensor: Updated point instance assignments, shape (S,)
    """
    t_start = time.time()

    normal_cos_thresh = math.cos(normal_angle_thresh/180.*np.pi)
    device = planes_normal.device

    if planes_ins_assignment_masked is None:
        assert precompued_plane_id is not None
        planes_ins_assignment_masked = precompued_plane_id.clone()
        unique_labels = precompued_plane_id.clone()
        mean_normal = planes_normal.clone()
        non_zero_mask = unique_labels != 0   # 0 means non plane
        used_labels = unique_labels[non_zero_mask]
        if len(used_labels) == 0:
            return planes_ins_assignment_masked.clone()
        max_label_num = len(used_labels)
    else:
        # 获取非零标签
        unique_labels = torch.unique(planes_ins_assignment_masked)
        non_zero_mask = unique_labels != 0   # 0 means non plane
        used_labels = unique_labels[non_zero_mask]
        if len(used_labels) == 0:
            return planes_ins_assignment_masked.clone()
        max_label_num = len(used_labels)
        # 创建标签映射: 真实标签 -> 连续索引 [0, M-1]
        label_to_idx = torch.zeros(used_labels.max().int().item() + 1, 
                                dtype=torch.long, device="cuda")
        label_to_idx[used_labels] = torch.arange(len(used_labels), device="cuda")
        # 获取每个点对应的索引 (背景点为-1)
        point_indices = label_to_idx[planes_ins_assignment_masked] 
        valid_mask = planes_ins_assignment_masked != 0
        
        # 向量化计算平均法向量 ================================
        if avg_type == 'mean':
            # 向量化均值计算
            sum_normals = torch.zeros((len(used_labels), 3), device="cuda")
            count_points = torch.zeros(len(used_labels), device="cuda")
            
            # 使用scatter_add进行高效聚合
            sum_normals.index_add_(0, point_indices[valid_mask], 
                                planes_normal[valid_mask])
            count_points.index_add_(0, point_indices[valid_mask], 
                                torch.ones_like(point_indices[valid_mask], dtype=torch.float))
            # 计算平均法向量并归一化
            mean_normal = sum_normals / count_points[:, None].clamp(min=1e-10)
            mean_normal = mean_normal / torch.norm(mean_normal, dim=1, keepdim=True).clamp(min=1e-10)
        elif avg_type == 'median':
            # 中位数计算仍需循环 (但仅对标签数量循环)
            mean_normal = torch.zeros((len(used_labels), 3), device="cuda")
            for i, label in enumerate(used_labels):
                mask = planes_ins_assignment_masked == label
                mean_normal[i] = torch.median(planes_normal[mask], dim=0)[0]
            mean_normal = mean_normal / torch.norm(mean_normal, dim=1, keepdim=True).clamp(min=1e-10)
        else:
            raise ValueError("avg_type must be 'mean' or 'median'")
    
    # calculate normal diff
    normal_diff_nxn = (mean_normal[None, :] * mean_normal[:, None]).sum(2)
    # calculate normal mask
    normal_mask_nxn = normal_diff_nxn > normal_cos_thresh
    normal_mask_nxn = normal_mask_nxn | normal_mask_nxn.t()

    final_mask = normal_mask_nxn
    final_mask.fill_diagonal_(False)

    all_idx = torch.arange(max_label_num).to(device)
    planes_ins_assignment_masked_tmp = planes_ins_assignment_masked.clone()

    t_preprocrss = time.time()

    t_label = 0.
    tmp_labels = used_labels.clone()
    while len(tmp_labels) > 0:
        label = tmp_labels[0]
        inlier_idx = all_idx[final_mask[label-1]] # idx in original labels
        t_tmp = time.time()
        if len(inlier_idx) > 0:
            # 优化实现：向量化操作
            old_labels = used_labels[inlier_idx] # find label in original labels

            mask = torch.isin(planes_ins_assignment_masked_tmp, old_labels)  # 创建布尔掩码
            planes_ins_assignment_masked_tmp[mask] = label  # 批量替换

            # record valid labels
            label_msk = torch.isin(tmp_labels, old_labels)
            label_msk[0] = True
            tmp_labels = tmp_labels[~label_msk]
        else:
            tmp_labels = tmp_labels[1:]
        t_label += time.time() - t_tmp

    # 7. 添加时间统计
    group_time = time.time() - t_preprocrss
    preprocess_time = t_preprocrss - t_start
    
    return planes_ins_assignment_masked_tmp

def get_continues_pts_ins_assignment(pts_plane_assignment: torch.Tensor) -> torch.Tensor:
    """
    Get continuous point instance assignments.
    
    Args:
        pts_plane_assignment (torch.Tensor): Point plane assignments, shape (N,)
        
    Returns:
        torch.Tensor: Continuous point plane assignments, shape (N,)
    """
    labels = pts_plane_assignment.unique()
    pts_plane_assignment_new = torch.zeros_like(pts_plane_assignment)
    new_label = 1
    for label in labels:
        if label > 0:
            pts_plane_assignment_new[pts_plane_assignment == label] = new_label
            new_label += 1
    # assert pts_plane_assignment_new.min() > 0
    return pts_plane_assignment_new

def save_pcd(points, labels, filename):
    """保存点云到PLY文件"""
    verts = points.cpu().numpy()
    verts[..., 1] *= -1  # 调整Y轴方向
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(verts)
    
    # 为每个标签生成随机颜色
    unique_labels = torch.unique(labels)
    random_colors = np.random.rand(unique_labels.max().int().item()+1, 3)
    colors = random_colors[labels.cpu().numpy()]
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(filename, pcd)

def upadte_plane_parameters(plane_center, plane_normal, plane_rot_q, plane_radii, plane_insIDs):
    if isinstance(plane_center, list):
        plane_center = torch.cat(plane_center, dim=0)
    if isinstance(plane_normal, list):
        plane_normal = torch.cat(plane_normal, dim=0)
    if isinstance(plane_rot_q, list):
        plane_rot_q = torch.cat(plane_rot_q, dim=0)
    if isinstance(plane_radii, list):
        plane_radii = torch.cat(plane_radii, dim=0)
    if isinstance(plane_insIDs, list):
        plane_insIDs = torch.cat(plane_insIDs, dim=0)
    
    unique_insID = torch.unique(plane_insIDs)
    plane_offset = -(plane_center * plane_normal).sum(dim=-1)
    plane_center_updated = plane_center.clone()
    plane_normal_updated = plane_normal.clone()
    plane_rot_q_updated = plane_rot_q.clone()
    for insID in unique_insID:
        if insID == 0:
            continue
        mask = plane_insIDs == insID
        masked_plane_normal = plane_normal[mask]
        masked_plane_radii = plane_radii[mask]
        masked_plane_centers = plane_center[mask]
        masked_plane_rot_q = plane_rot_q[mask]

        masked_plane_size = masked_plane_radii[:, 0] * masked_plane_radii[:, 1]
        weight = masked_plane_size / torch.sum(masked_plane_size)
        avg_plane_offset = torch.sum(weight * plane_offset[mask])
        avg_plane_normal = torch.sum(weight[:, None] * masked_plane_normal, dim=0)
        avg_plane_normal = avg_plane_normal / torch.norm(avg_plane_normal)

        # 计算投影点：P_proj = P - (n·P + d)*n
        # 计算点积 (n·P)
        dot_product = torch.sum(masked_plane_centers * avg_plane_normal, dim=1)
        # 计算投影点
        projections = masked_plane_centers - ((dot_product + avg_plane_offset)[:, None] * avg_plane_normal)
        
        plane_center_updated[mask] = projections
        plane_normal_updated[mask] = avg_plane_normal

        # update plane rotation
        ## calculate rot between original normal and updated normal
        angle_diff = torch.acos((masked_plane_normal * avg_plane_normal[None]).sum(dim=-1).clamp(-1, 1)).reshape(-1, 1)
        rot_axis = torch.cross(masked_plane_normal, avg_plane_normal[None], dim=-1)  # n_plane, 3
        rot_axis = F.normalize(rot_axis, dim=-1)
        rot_vec = (rot_axis * angle_diff).cpu().numpy()
        rot_q_delta = quaternion.as_float_array(quaternion.from_rotation_vector(rot_vec))
        rot_q_delta = torch.from_numpy(rot_q_delta).float().cuda()

        rot_q_updated = quaternion_mult(rot_q_delta, masked_plane_rot_q)
        plane_rot_q_updated[mask] = rot_q_updated

    return plane_center_updated, plane_normal_updated, plane_rot_q_updated, plane_radii

def merge_plane(
    plane_normal: torch.Tensor,  # n, 3
    plane_center: torch.Tensor,  # n, 3
    plane_radii: torch.Tensor,   # n, 2
    plane_rot_q: torch.Tensor,   # n, 4
    normal_angle_thresh: float = 25,
    space_resolution: float = 0.02,
    cc_dist_thresh: float = 0.02,
    cc_delta: float = 0.02,
    area_thresh: float = 0.16,
    area_thresh2: float = 0.16,
) -> tuple:
    """
    Merge 3D plane primitives into a consolidated point cloud representation.
    
    This function takes a set of 3D plane primitives (defined by their normal, center, radii, and rotation)
    and processes them through several steps to create a merged point cloud with instance assignments.
    
    """
    torch.use_deterministic_algorithms(False)    

    # Get parameters of 3D plane primitives
    plane_normal = plane_normal.detach()
    plane_center = plane_center.detach()
    plane_radii = plane_radii.detach()
    plane_rot_q = plane_rot_q.detach()

    # get plane corners
    plane_normals_standard = torch.zeros_like(plane_normal)
    plane_normals_standard[..., -1] = 1
    if plane_radii.shape[-1] == 2:
        radii_x_p = plane_radii[..., 0]  # n
        radii_y_p = plane_radii[..., 1]  # n
        radii_x_n = plane_radii[..., 0]  # n
        radii_y_n = plane_radii[..., 1]  # n
    elif plane_radii.shape[-1] == 4:
        radii_x_p = plane_radii[..., 0]  # n
        radii_y_p = plane_radii[..., 1]  # n
        radii_x_n = plane_radii[..., 2]  # n
        radii_y_n = plane_radii[..., 3]  # n
    else:
        raise NotImplementedError
    zero_tmp = torch.zeros_like(radii_x_p)  # n
    v1 = torch.stack([radii_x_p, radii_y_p, zero_tmp], dim=-1)  # n, 3
    v2 = torch.stack([-radii_x_n, radii_y_p, zero_tmp], dim=-1)  # n, 3
    v3 = torch.stack([-radii_x_n, -radii_y_n, zero_tmp], dim=-1)  # n, 3
    v4 = torch.stack([radii_x_p, -radii_y_n, zero_tmp], dim=-1)  # n, 3
    vertices_standard = torch.stack([v1, v2, v3, v4], dim=1)  # n, 4, 3
    plane_rot_q = F.normalize(plane_rot_q, dim=-1)  # n, 4
    rot_matrix = quat_to_rot(plane_rot_q)  # n, 3, 3
    plane_corner = torch.bmm(rot_matrix, vertices_standard.permute(0, 2, 1)).permute(0, 2, 1) + plane_center[:, None]  # n, 4, 3

    # Sample points from the 3D plane primitives
    plane_id = torch.arange(plane_center.shape[0]).cuda() + 1
    
    planes_normal_original = plane_normal.clone()
       
    # Split planes into different groups via normal
    planes_ins_assignment_masked_NG = group_plane_via_normal_vectorized(
        planes_normal=planes_normal_original,
        # planes_ins_assignment_masked=planes_ins_assignment_masked,
        planes_ins_assignment_masked=None,
        normal_angle_thresh=normal_angle_thresh,
        precompued_plane_id=plane_id
    )

    planes_ins_assignment_masked_NG = get_continues_pts_ins_assignment(planes_ins_assignment_masked_NG).int()

    # Check connected components
    planes_ins_assignment_masked_NG_cc = torch.zeros_like(planes_ins_assignment_masked_NG)
    NG_ids = planes_ins_assignment_masked_NG.unique()
    NG_ids = NG_ids[NG_ids != 0]
    for ngid in NG_ids:         
        ngid_group_mask = planes_ins_assignment_masked_NG == ngid
        plane_normal_group = planes_normal_original[ngid_group_mask]
        plane_corner_group = plane_corner[ngid_group_mask]
        plane_center_group = plane_center[ngid_group_mask]
        plane_ins_assignment_group = check_cc_plane(
            plane_center=plane_center_group,
            plane_normal=plane_normal_group,
            plane_pts_dense=plane_corner_group,
            dist_thresh = cc_dist_thresh,
            delta = cc_delta,
            area_thresh = area_thresh,
            area_thresh2= area_thresh2
        ) # 0: non-plane
        last_id = planes_ins_assignment_masked_NG_cc.max()
        plane_ins_assignment_group[plane_ins_assignment_group > 0] += last_id + 1
        planes_ins_assignment_masked_NG_cc[ngid_group_mask] = plane_ins_assignment_group
    planes_ins_assignment_masked_NG_cc = get_continues_pts_ins_assignment(planes_ins_assignment_masked_NG_cc).int()
    # remove small plane
    valid_mask = planes_ins_assignment_masked_NG_cc != 0
    
    # update plane parameters
    plane_center_updated, plane_normal_updated, plane_rot_q_updated, _ = upadte_plane_parameters(
            plane_center, plane_normal, plane_rot_q, plane_radii, planes_ins_assignment_masked_NG_cc)
    # get updated plane corners
    plane_rot_q_updated = F.normalize(plane_rot_q_updated, dim=-1)  # n, 4
    rot_matrix_updated = quat_to_rot(plane_rot_q_updated)  # n, 3, 3
    plane_corner_updated = torch.bmm(rot_matrix_updated, vertices_standard.permute(0, 2, 1)).permute(0, 2, 1) + plane_center_updated[:, None]  # n, 4, 3
    #  group via normal
    planes_ins_assignment_masked_NG = group_plane_via_normal_vectorized(
            planes_normal=plane_normal_updated,
            planes_ins_assignment_masked=planes_ins_assignment_masked_NG_cc,
            normal_angle_thresh=normal_angle_thresh
            )
    planes_ins_assignment_masked_NG = get_continues_pts_ins_assignment(planes_ins_assignment_masked_NG).int()
    # Check connected components
    planes_ins_assignment_masked_NG_cc = torch.zeros_like(planes_ins_assignment_masked_NG)
    NG_ids = planes_ins_assignment_masked_NG.unique()
    NG_ids = NG_ids[NG_ids != 0]
    for ngid in NG_ids:         
        ngid_group_mask = planes_ins_assignment_masked_NG == ngid
        plane_normal_group = plane_normal_updated[ngid_group_mask]
        plane_corner_group = plane_corner_updated[ngid_group_mask]
        plane_center_group = plane_center_updated[ngid_group_mask]
        plane_ins_assignment_group = check_cc_plane(
            plane_center=plane_center_group,
            plane_normal=plane_normal_group,
            plane_pts_dense=plane_corner_group,
            dist_thresh = min(0.02, cc_dist_thresh),
            delta = cc_delta,
            area_thresh = area_thresh,
            area_thresh2= area_thresh2
        ) # 0: non-plane
        last_id = planes_ins_assignment_masked_NG_cc.max()
        plane_ins_assignment_group[plane_ins_assignment_group > 0] += last_id + 1
        planes_ins_assignment_masked_NG_cc[ngid_group_mask] = plane_ins_assignment_group
    planes_ins_assignment_masked_NG_cc = get_continues_pts_ins_assignment(planes_ins_assignment_masked_NG_cc).int()
    # remove small plane
    valid_mask = planes_ins_assignment_masked_NG_cc != 0

    plane_ins_num = len(planes_ins_assignment_masked_NG_cc[valid_mask].unique())
    print(f'plane ins num = {plane_ins_num}')

    return planes_ins_assignment_masked_NG_cc, valid_mask