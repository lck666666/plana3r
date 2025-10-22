import os
import math
from typing import List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from tqdm import tqdm
import random
import open3d as o3d
import itertools
from pytorch3d.ops import knn_points
from typing import List
import time

def sample_pcd_from_prims(
    plane_normal, # n, 3
    plane_center, # n, 3
    plane_radii,  # n, 2
    plane_rot_q,  # n, 4
    space_resolution: float=0.02,
    ):
    torch.use_deterministic_algorithms(False)
    ## get parameters of 3D plane primitives
    plane_normal = plane_normal.detach()
    plane_center = plane_center.detach()
    plane_radii = plane_radii.detach()
    plane_rot_q = plane_rot_q.detach()

    ## sample points from the 3D plane primitives
    plane_ins_id = torch.arange(plane_center.shape[0]).cuda() + 1
    pts_original, pts_ins_assignment_original, pts_normal_original, faces_original = sample_pts_from_GivenPlanePrim(
        plane_normal, 
        plane_center, 
        plane_radii, 
        plane_rot_q, 
        space_resolution=space_resolution,
        plane_ins_id=plane_ins_id)
    
    return pts_original

def sample_one_prim(radii_x_i, radii_y_i, sample_interval, rot_matrix_i, plane_normal_i, plane_ins_id_i, plane_center_i):
    rx, ry = radii_x_i.item(), radii_y_i.item()
    nx, ny = int(max(2 * rx // sample_interval + 1, 3)), int(max(2 * ry // sample_interval + 1, 3))

    px = torch.linspace(-rx, rx, nx).reshape(-1, 1, 1).repeat(1, ny, 1).cuda()
    py = torch.linspace(-ry, ry, ny).reshape(1, -1, 1).repeat(nx, 1, 1).cuda()
    pz = torch.zeros_like(py)
    pts_i = torch.cat([px, py, pz], dim=-1).reshape(-1, 3)
    pts_i_transformed = torch.mm(rot_matrix_i, pts_i.permute(1, 0)).permute(1, 0) + plane_center_i.reshape(1, 3)

    normal_i = plane_normal_i.reshape(1, 3).repeat(pts_i_transformed.shape[0], 1)
    pts_planeID = plane_ins_id_i * torch.ones(pts_i_transformed.shape[0]).to(pts_i_transformed.device)

    return (normal_i, pts_i_transformed, pts_planeID.int())

def sample_pts_from_GivenPlanePrim_fast(
        plane_normal, 
        plane_center, 
        plane_radii, 
        plane_rot_q, 
        scene_scale=1.0,
        scene_offset=[0., 0., 0.],
        space_resolution=0.05, 
        plane_ins_id=None
        ):
    """
    space_resolution=0.05 --> 0.05 meter
    """
    sample_interval = space_resolution * scene_scale  # to normalized space

    rot_q = F.normalize(plane_rot_q, dim=-1)  # n, 4
    rot_matrix = quat_to_rot(rot_q)  # n, 3, 3
    plane_normal_standard = torch.zeros_like(plane_normal)
    plane_normal_standard[..., -1] = 1
    radii_x = plane_radii[..., 0]  # n
    radii_y = plane_radii[..., 1]  # n

    if plane_ins_id is None:
        plane_ins_id = torch.arange(plane_center.shape[0]) + 1 # valid plane ins IDs start from 1
    plane_ins_id = plane_ins_id.reshape(-1).int()  # 0 means non-plane ins
            
    pts = []
    per_point_plane_assignment = []
    normals = []

    for i in tqdm(range(len(plane_normal)), desc='sampleing points from planes...'):
        normal_i, pts_i_transformed, pts_planeID = sample_one_prim(radii_x[i], radii_y[i], sample_interval, rot_matrix[i], plane_normal[i], plane_ins_id[i], plane_center[i])
        normals.append(normal_i)
        pts.append(pts_i_transformed)
        per_point_plane_assignment.append(pts_planeID.int())

    # import concurrent.futures
    # params_list = [
    #     (radii_x[i], radii_y[i], sample_interval, rot_matrix[i], plane_normal[i], plane_ins_id[i], plane_center[i])
    #     for i in range(len(radii_x))
    # ]
    # with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
    #     futures = []
    #     # 提交所有任务
    #     for params in params_list:
    #         futures.append(
    #             executor.submit(
    #                 sample_one_prim,  # 修改后的线程安全版本
    #                 *params
    #             )
    #         )
    #     # 使用tqdm显示进度
    #     for future in tqdm(concurrent.futures.as_completed(futures), 
    #                      total=len(futures),
    #                      desc="Parallel sampling"):
    #         normal_i, pts_i, plane_id = future.result()
    #         normals.append(normal_i)
    #         pts.append(pts_i)
    #         per_point_plane_assignment.append(plane_id)

    pts = torch.cat(pts, dim=0).detach()
    pts /= scene_scale
    scene_offset = torch.tensor(scene_offset).cuda().reshape(1, 3)
    pts += scene_offset
    normals = torch.cat(normals, dim=0).detach()
    per_point_plane_assignment = torch.cat(per_point_plane_assignment, dim=0).detach()

    return pts, per_point_plane_assignment, normals

def merge_plane_fast_pcd(
        plane_normal, # n, 3
        plane_center, # n, 3
        plane_radii,  # n, 2
        plane_rot_q,  # n, 4
        coarse_mesh_o3d = None,
        mesh_dist_thresh=0.02,
        plane_ins_id: Optional[List]=None, 
        normal_angle_thresh: float=25,
        dist_thresh: float=0.05, 
        floor_height: float=-1,
        ceiling_height: float=-1,
        space_resolution: float=0.02,
        voxel_size: float=0.01,
        fc_voxel_size: float=0.05,
        find_fc_normal_angle_thresh: float=25,
        fc_normal_angle_thresh: float=25,
        fc_dist_thresh: float=0.2,
        return_ins_parameters: bool=False
        ):
    torch.use_deterministic_algorithms(False)
    min_pts_num = max((0.1/space_resolution)**2, 200)

    ## get parameters of 3D plane primitives
    plane_normal = plane_normal.detach()
    plane_center = plane_center.detach()
    plane_radii = plane_radii.detach()
    plane_rot_q = plane_rot_q.detach()

    ## sample points from the 3D plane primitives
    if plane_ins_id is None:
        plane_ins_id = torch.arange(plane_center.shape[0]).cuda() + 1
    # import time
    # time_s1 = time.time()
    pts_original, pts_ins_assignment_original, pts_normal_original = sample_pts_from_GivenPlanePrim_fast(
        plane_normal, 
        plane_center, 
        plane_radii, 
        plane_rot_q, 
        space_resolution=space_resolution,
        plane_ins_id=plane_ins_id)
    # time_s2 = time.time()
    # print(time_s2 - time_s1)
    # import pdb; pdb.set_trace()
    
    ## calculate masked pts assignment
    pts_ins_assignment_masked = pts_ins_assignment_original.clone()

    if coarse_mesh_o3d is not None:
        ## calculate pts2mesh distance
        dist_pts2mesh_original = calculate_pts2mesh_dist(pts_original, coarse_mesh_o3d)
        pts_ins_assignment_masked[dist_pts2mesh_original > mesh_dist_thresh] = 0
    ## split planes into different group via normal 
    pts_ins_assignment_masked_NG = group_plane_via_normal(
        pts_normal_original, # S, 3
        pts_ins_assignment_masked, # S
        normal_angle_thresh=normal_angle_thresh,
    )
    pts_ins_assignment_masked_NG = get_continues_pts_ins_assignment(pts_ins_assignment_masked_NG).int()
    ## update pts' normal
    pts_normal_updated = update_pts_normal(pts_ins_assignment_masked_NG, pts_normal_original, use_median=True)
    ## adjust pts for each plane (***optinal***)
    # if True:
    if False:
        pts_updated = move_pts_onto_plane_simple(pts_original.clone(), pts_normal_updated, pts_ins_assignment_masked)
    else:
        pts_updated = pts_original.clone()

    ## check cc
    pts_ins_assignment_masked_NG_cc = torch.zeros_like(pts_ins_assignment_masked_NG)
    NG_ids = pts_ins_assignment_masked_NG.unique()
    for ngid in NG_ids:         
        if ngid == 0:
            continue   
        gmask = pts_ins_assignment_masked_NG == ngid
        pts_ins_assignment_masked_tmp = pts_ins_assignment_masked * gmask.float()
        pts_ins_assignment_masked_tmp = get_continues_pts_ins_assignment(pts_ins_assignment_masked_tmp).int()
        pts_ins_assignment_masked_tmp_cc = check_cc(
            pts_updated, # S, 3
            # pts_original, # S, 3
            pts_normal_updated, # S, 3
            # pts_normal_original, # S, 3
            pts_ins_assignment_masked_tmp, # S
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            adj_ratio_threshold=0.0, 
            adj_count_threshold=5,
            voxel_size=voxel_size,
            fc_adj_ratio_threshold=0.0, 
            fc_adj_count_threshold=5,
            fc_voxel_size=fc_voxel_size,
            find_fc_normal_angle_thresh=find_fc_normal_angle_thresh,
        )
        last_id = pts_ins_assignment_masked_NG_cc.max() + 1
        pts_ins_assignment_masked_NG_cc[gmask] = pts_ins_assignment_masked_tmp_cc[gmask] + last_id
    pts_ins_assignment_masked_NG_cc= get_continues_pts_ins_assignment(pts_ins_assignment_masked_NG_cc).int()

    ## check distdance
    pts_ins_assignment_masked_tmp = torch.zeros_like(pts_ins_assignment_masked)
    start_ins_id = 1
    for group_id in pts_ins_assignment_masked_NG_cc.unique():
        if group_id == 0:
            continue
        pts_ins_assignment_masked_cur = find_best_assignments_for_each_group(
            group_id, 
            pts_updated,
            pts_normal_updated, 
            pts_ins_assignment_masked_NG_cc, 
            pts_ins_assignment_masked, 
            start_ins_id,
            dist_thres=dist_thresh)
        assert pts_ins_assignment_masked_tmp[pts_ins_assignment_masked_cur>0].sum() == 0
        pts_ins_assignment_masked_tmp = pts_ins_assignment_masked_tmp + pts_ins_assignment_masked_cur
        start_ins_id = pts_ins_assignment_masked_tmp.max()+1
    pts_ins_assignment_masked_NG_cc_DG = get_continues_pts_ins_assignment(pts_ins_assignment_masked_tmp).int()

    ## merge floor again
    # pts_ins_assignment_masked_NG_cc_DG_mf = merge_fc_plane(
    #     pts_updated, # S, 3
    #     pts_normal_updated, # S, 3
    #     pts_ins_assignment_masked_NG_cc_DG, # S
    #     fc_normal_angle_thresh=fc_normal_angle_thresh,
    #     fc_dist_thresh=fc_dist_thresh,
    #     find_fc_normal_angle_thresh=find_fc_normal_angle_thresh,
    #     floor_height=floor_height,
    #     ceiling_height=-1,
    #     fc_adj_ratio_threshold=0., 
    #     fc_adj_count_threshold=1,
    #     fc_voxel_size=fc_voxel_size
    # )
    # pts_ins_assignment_masked_NG_cc_DG_mf = get_continues_pts_ins_assignment(pts_ins_assignment_masked_NG_cc_DG_mf).int()
    pts_ins_assignment_masked_NG_cc_DG_mf = pts_ins_assignment_masked_NG_cc_DG

    pts_ins_assignment_final = pts_ins_assignment_masked_NG_cc_DG_mf.clone()

    ## move points onto plane
    pts_updated = move_pts_onto_plane_simple(pts_updated.clone(), pts_normal_updated, pts_ins_assignment_final)

    ## remove small planes
    min_ins_pts = min_pts_num
    count = 0
    for label in pts_ins_assignment_final.unique():
        if label == 0:
            continue
        mask = pts_ins_assignment_final == label
        if mask.sum() < min_ins_pts:
            pts_ins_assignment_final[mask] = 0
        else:
            count += 1
    logger.info("number of planar instances = %d"%(count))
    # pts_ins_assignment_final = get_continues_pts_ins_assignment(pts_ins_assignment_final).int()

    mask = pts_ins_assignment_final > 0
    planar_pcd = pts_updated[mask]
    pts_ins_assignment_final = pts_ins_assignment_final[mask]

    # import pdb; pdb.set_trace()
    return planar_pcd, pts_ins_assignment_final

def merge_plane(
        plane_normal, # n, 3
        plane_center, # n, 3
        plane_radii,  # n, 2
        plane_rot_q,  # n, 4
        coarse_mesh_o3d = None,
        mesh_dist_thresh=0.02,
        plane_ins_id: Optional[List]=None, 
        normal_angle_thresh: float=25,
        dist_thresh: float=0.05, 
        floor_height: float=-1,
        ceiling_height: float=-1,
        space_resolution: float=0.02,
        voxel_size: float=0.01,
        fc_voxel_size: float=0.05,
        find_fc_normal_angle_thresh: float=25,
        fc_normal_angle_thresh: float=25,
        fc_dist_thresh: float=0.2,
        return_ins_parameters: bool=False,
        colorMap_vis=None,
        adj_count_threshold=5,
        ):
    torch.use_deterministic_algorithms(False)
    min_pts_num = max((0.1/space_resolution)**2, 200)

    ## get parameters of 3D plane primitives
    plane_normal = plane_normal.detach()
    plane_center = plane_center.detach()
    plane_radii = plane_radii.detach()
    plane_rot_q = plane_rot_q.detach()

    ## sample points from the 3D plane primitives
    if plane_ins_id is None:
        plane_ins_id = torch.arange(plane_center.shape[0]).cuda() + 1
    
    time_s1 = time.time()
    pts_original, pts_ins_assignment_original, pts_normal_original, faces_original = sample_pts_from_GivenPlanePrim(
        plane_normal, 
        plane_center, 
        plane_radii, 
        plane_rot_q, 
        space_resolution=space_resolution,
        plane_ins_id=plane_ins_id)
    time_s2 = time.time()
    # print('Time -- sample_pts_from_GivenPlanePrim:  ', time_s2 - time_s1)
    
    ## calculate masked pts assignment
    pts_ins_assignment_masked = pts_ins_assignment_original.clone()

    if coarse_mesh_o3d is not None:
        ## calculate pts2mesh distance
        dist_pts2mesh_original = calculate_pts2mesh_dist(pts_original, coarse_mesh_o3d)
        pts_ins_assignment_masked[dist_pts2mesh_original > mesh_dist_thresh] = 0
    time_s3 = time.time()
    # print('Time -- calculate_pts2mesh_dist:  ', time_s3 - time_s2)

    ## split planes into different group via normal 
    pts_ins_assignment_masked_NG = group_plane_via_normal(
        pts_normal_original, # S, 3
        pts_ins_assignment_masked, # S
        normal_angle_thresh=normal_angle_thresh,
    )
    time_s4 = time.time()
    # print('Time -- group_plane_via_normal:  ', time_s4 - time_s3)

    pts_ins_assignment_masked_NG = get_continues_pts_ins_assignment(pts_ins_assignment_masked_NG).int()
    ## update pts' normal
    pts_normal_updated = update_pts_normal(pts_ins_assignment_masked_NG, pts_normal_original, use_median=True)

    time_s5 = time.time()
    # print('Time -- update_pts_normal:  ', time_s5 - time_s4)

    ## adjust pts for each plane (***optinal***)
    if True:
        pts_updated = move_pts_onto_plane_simple(pts_original.clone(), pts_normal_updated, pts_ins_assignment_masked)
    else:
        pts_updated = pts_original.clone()
    
    time_s6 = time.time()
    # print('Time -- move_pts_onto_plane_simple:  ', time_s6 - time_s5)
    
    ## check cc
    pts_ins_assignment_masked_NG_cc = torch.zeros_like(pts_ins_assignment_masked_NG)
    NG_ids = pts_ins_assignment_masked_NG.unique()
    for ngid in NG_ids:         
        if ngid == 0:
            continue   
        gmask = pts_ins_assignment_masked_NG == ngid
        pts_ins_assignment_masked_tmp = pts_ins_assignment_masked * gmask.float()
        pts_ins_assignment_masked_tmp = get_continues_pts_ins_assignment(pts_ins_assignment_masked_tmp).int()
        pts_ins_assignment_masked_tmp_cc = check_cc(
            pts_updated, # S, 3
            # pts_original, # S, 3
            pts_normal_updated, # S, 3
            # pts_normal_original, # S, 3
            pts_ins_assignment_masked_tmp, # S
            floor_height=floor_height,
            ceiling_height=ceiling_height,
            adj_ratio_threshold=0.0, 
            adj_count_threshold=10, # 5
            voxel_size=voxel_size,
            fc_adj_ratio_threshold=0.0, 
            fc_adj_count_threshold=adj_count_threshold,
            fc_voxel_size=fc_voxel_size,
            find_fc_normal_angle_thresh=find_fc_normal_angle_thresh,
        )
        last_id = pts_ins_assignment_masked_NG_cc.max() + 1
        pts_ins_assignment_masked_NG_cc[gmask] = pts_ins_assignment_masked_tmp_cc[gmask] + last_id
    pts_ins_assignment_masked_NG_cc= get_continues_pts_ins_assignment(pts_ins_assignment_masked_NG_cc).int()

    time_s7 = time.time()
    # print('Time -- check cc:  ', time_s7 - time_s6)

    ## check distdance
    pts_ins_assignment_masked_tmp = torch.zeros_like(pts_ins_assignment_masked)
    start_ins_id = 1
    for group_id in pts_ins_assignment_masked_NG_cc.unique():
        if group_id == 0:
            continue
        pts_ins_assignment_masked_cur = find_best_assignments_for_each_group(
            group_id, 
            pts_updated,
            pts_normal_updated, 
            pts_ins_assignment_masked_NG_cc, 
            pts_ins_assignment_masked, 
            start_ins_id,
            dist_thres=dist_thresh)
        assert pts_ins_assignment_masked_tmp[pts_ins_assignment_masked_cur>0].sum() == 0
        pts_ins_assignment_masked_tmp = pts_ins_assignment_masked_tmp + pts_ins_assignment_masked_cur
        start_ins_id = pts_ins_assignment_masked_tmp.max()+1
    pts_ins_assignment_masked_NG_cc_DG = get_continues_pts_ins_assignment(pts_ins_assignment_masked_tmp).int()

    time_s8 = time.time()
    # print('Time -- find_best_assignments_for_each_group:  ', time_s8 - time_s7)

    ## merge floor again
    pts_ins_assignment_masked_NG_cc_DG_mf = merge_fc_plane(
        pts_updated, # S, 3
        pts_normal_updated, # S, 3
        pts_ins_assignment_masked_NG_cc_DG, # S
        fc_normal_angle_thresh=fc_normal_angle_thresh,
        fc_dist_thresh=fc_dist_thresh,
        find_fc_normal_angle_thresh=find_fc_normal_angle_thresh,
        floor_height=floor_height,
        ceiling_height=-1,
        fc_adj_ratio_threshold=0., 
        fc_adj_count_threshold=1,
        fc_voxel_size=fc_voxel_size
    )
    pts_ins_assignment_masked_NG_cc_DG_mf = get_continues_pts_ins_assignment(pts_ins_assignment_masked_NG_cc_DG_mf).int()

    time_s9 = time.time()
    # print('Time -- merge_fc_plane:  ', time_s9 - time_s8)


    pts_ins_assignment_final = pts_ins_assignment_masked_NG_cc_DG_mf.clone()
    ''''''
    ## move points onto plane
    pts_updated = move_pts_onto_plane_simple(pts_updated.clone(), pts_normal_updated, pts_ins_assignment_final)
    ''''''

    time_s10 = time.time()
    # print('Time -- move_pts_onto_plane_simple:  ', time_s10 - time_s9)

    ## remove small planes
    min_ins_pts = min_pts_num
    count = 0
    for label in pts_ins_assignment_final.unique():
        if label == 0:
            continue
        mask = pts_ins_assignment_final == label
        if mask.sum() < min_ins_pts:
            pts_ins_assignment_final[mask] = 0
        else:
            count += 1
    logger.info("number of planar instances = %d"%(count))
    pts_ins_assignment_final = get_continues_pts_ins_assignment(pts_ins_assignment_final).int()

    time_s11 = time.time()
    # print('Time -- remove small planes:  ', time_s11 - time_s10)

    ''''''
    plane_ins_id_new = get_planeInsId_from_ptsInsAssignment(plane_normal.shape[0], pts_ins_assignment_original, pts_ins_assignment_final)
    ''''''

    time_s12 = time.time()
    # print('Time -- get_planeInsId_from_ptsInsAssignment:  ', time_s12 - time_s11)

    planar_mesh, mesh_pts = build_planar_mesh_for_eval(
        pts_updated, 
        pts_normal_updated, 
        pts_ins_assignment_final, 
        faces_original, 
        return_pts=True,
        colorMap_vis=colorMap_vis)
    
    time_s13 = time.time()
    # print('Time -- build_planar_mesh_for_eval:  ', time_s13 - time_s12)
    
    # import pdb; pdb.set_trace()
    if return_ins_parameters:
        unique_ins_ids = plane_ins_id_new.unique()  # Note: 0 means non-plane ins
        ins_normals = []
        ins_offsets = []
        ins_pts = []
        for ins_id in unique_ins_ids:
            if ins_id == 0:
                continue
            mask = pts_ins_assignment_final == ins_id   
            sample_pts = mesh_pts[mask]     
            plane_normal = torch.median(pts_normal_updated[mask], dim=0)[0]
            plane_offset = -torch.median((sample_pts @ plane_normal))

            ins_normals.append(plane_normal)
            ins_offsets.append(plane_offset)
            ins_pts.append(sample_pts)
        return planar_mesh, plane_ins_id_new, ins_pts, ins_normals, ins_offsets
    else:
        return planar_mesh, plane_ins_id_new

def quat_to_rot(q):
    assert isinstance(q, torch.Tensor)
    assert q.shape[-1] == 4
    if q.dim() == 1:
        q = q.unsqueeze(0)  # 1, 4
    elif q.dim() == 2:
        pass  # bs, 4
    else:
        raise NotImplementedError

    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    # R = torch.zeros((batch_size, 3, 3), device='cuda')
    qr = q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0] = 1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R

def merge_fc_plane(
    pts, # S, 3
    pts_normal, # S, 3
    pts_ins_assignment_masked, # S
    fc_normal_angle_thresh=25,
    fc_dist_thresh=0.20,
    find_fc_normal_angle_thresh=25,
    floor_height=0.2,
    ceiling_height=2.0,
    fc_adj_ratio_threshold=0., 
    fc_adj_count_threshold=1,
    fc_voxel_size=0.1
):
    normal_cos_thresh = math.cos(15/180.*np.pi)
    dist_thresh = 0.05
    voxel_size = 0.02

    fc_normal_cos_thresh = math.cos(fc_normal_angle_thresh/180.*np.pi)
    find_fc_normal_cos_thresh = math.cos(find_fc_normal_angle_thresh/180.*np.pi)
    unique_labels = pts_ins_assignment_masked.unique()

    max_label_num = len(unique_labels)
    if unique_labels[0] == 0:
        max_label_num -= 1
    mean_normal = torch.zeros((max_label_num, 3), device="cuda")
    mean_offset = torch.zeros((max_label_num, 1), device="cuda")
    mean_z = torch.zeros((max_label_num, 1), device="cuda")
    mean_proj_dist = torch.ones((max_label_num, max_label_num), device="cuda") * 100.
    used_labels = []
    pts_list = []
    pts_num = []

    count = 0
    for label in tqdm(unique_labels, total=len(unique_labels)):
        if label == 0:
            continue
        mask = pts_ins_assignment_masked == label
        assert mask.sum() > 0
        used_labels.append(label)
        # compute plane parameters and move all points onto the plane
        plane_pts = pts[mask]
        plane_normal = torch.median(pts_normal[mask], dim=0)[0]
        # plane_normal = torch.mean(original_normals_array[mask], dim=0)
        normal = plane_normal / (torch.norm(plane_normal) + 1e-10)

        # offset = -torch.median((plane_points * normal[None, :]).sum(-1), dim=0)[0]
        offset = -torch.mean((plane_pts * normal[None, :]).sum(-1), dim=0)

        z = torch.mean(plane_pts, dim=0)[-1]

        mean_normal[count] = normal
        mean_offset[count] = offset
        mean_z[count] = z

        pts_list.append(plane_pts.clone())
        pts_num.append(mask.sum())

        count += 1
    
    assert len(used_labels) == count
    if count == 1:
        pts_ins_assignment_masked_updated = torch.zeros_like(pts_ins_assignment_masked)
        pts_ins_assignment_masked_updated[pts_ins_assignment_masked==used_labels[0]] = 1
        return pts_ins_assignment_masked_updated
    try:
        pts_num = torch.stack(pts_num).squeeze()
    except:
        import pdb; pdb.set_trace()

    count = 0
    for label in tqdm(unique_labels, total=len(unique_labels)):
        if label == 0:
            continue
        mask = pts_ins_assignment_masked == label
        assert mask.sum() > 0
        # compute plane parameters and move all points onto the plane
        plane_pts = pts[mask]

        dist = (plane_pts.reshape(1, -1, 3) * mean_normal.reshape(-1, 1, 3)).sum(-1) + mean_offset.reshape(-1, 1)
        m_dist = dist.abs().mean(dim=-1)
        mean_proj_dist[count] = m_dist
        count += 1

    assert len(used_labels) == count
    
    # calculate normal diff and dist diff
    normal_diff_nxn = (mean_normal[None, :] * mean_normal[:, None]).sum(2)
    proj_dist_diff_nxn = mean_proj_dist

    # calculate normal and dist mask
    normal_mask_nxn = ((normal_diff_nxn > normal_cos_thresh))
    normal_mask_nxn = normal_mask_nxn | normal_mask_nxn.t()
    dist_mask_nxn = (proj_dist_diff_nxn < dist_thresh)
    dist_mask_nxn = dist_mask_nxn | dist_mask_nxn.t()

    # calculate overlap mask
    adj_mask_nxn = calculate_adj_mask(pts_list, adj_ratio_threshold=0, adj_count_threshold=1, voxel_size=0.02)
    adj_mask_nxn = adj_mask_nxn | adj_mask_nxn.t()

    adj_mask_nxn_fc = calculate_adj_mask(pts_list, fc_adj_ratio_threshold, fc_adj_count_threshold, fc_voxel_size)
    adj_mask_nxn_fc = adj_mask_nxn_fc | adj_mask_nxn_fc.t()

    device = normal_diff_nxn.device
    mean_z = mean_z.squeeze()
    vec_z = torch.tensor([0,0,1]).to(device).reshape(-1, 3) * mean_z.reshape(-1, 1)  # n, 3

    if floor_height > 0:
        # get floor mask coarse
        floor_normal = torch.tensor([0,0,1]).to(device)
        floor_normal_diff = (mean_normal * floor_normal.reshape(1, 3)).sum(dim=-1)
        floor_normal_mask =  floor_normal_diff > find_fc_normal_cos_thresh

        if floor_normal_mask.sum() < 10:
            floor_mask = torch.zeros_like(floor_normal_mask)
        else:
            min_z = mean_z[floor_normal_mask].min().cpu().item()
            for i in range(10):
                floor_z_mask = mean_z < min_z + floor_height + 0.01 * i
                floor_mask = floor_normal_mask & floor_z_mask
                if floor_mask.sum() > 10:
                    break
        if floor_mask.sum() > 10:
            floor_min_z = mean_z[floor_mask].min().cpu().item()
            # floor_min_z = mean_z[floor_mask].mean().cpu().item()

            # get floor mask fine
            ## calculate new floor normal
            floor_prim_weight = pts_num[floor_mask] / pts_num[floor_mask].sum()
            floor_normal = F.normalize((mean_normal[floor_mask] * floor_prim_weight.reshape(-1, 1)).sum(dim=0), dim=-1)
            ## calculate new floor normal mask
            floor_normal_diff = (mean_normal * floor_normal.reshape(1, 3)).sum(dim=-1)
            floor_normal_mask =  floor_normal_diff > find_fc_normal_cos_thresh
            ## we should align z to new floor normal !!!
            mean_z_fn_adjusted = (vec_z * floor_normal.reshape(1, 3)).sum(dim=-1)
            floor_min_z_fn_adjusted = (torch.tensor([0,0,floor_min_z]).to(device).reshape(1, 3) * floor_normal.reshape(1, 3)).sum(dim=-1).cpu().item()
            ## calculate new floor z mask
            # floor_z_mask = mean_z < floor_min_z + floor_height
            floor_z_mask = mean_z_fn_adjusted < floor_min_z_fn_adjusted + floor_height
            ## get pairwise floor mask
            floor_mask = floor_normal_mask & floor_z_mask
            floor_mask_nxn = (floor_mask[None, :] * floor_mask[:, None])

            # update normal mask
            floor_mask_pad = floor_mask[:, None].repeat(1, floor_mask.shape[0])
            normal_mask_nxn[floor_mask_pad] = 0
            normal_mask_nxn = normal_mask_nxn & normal_mask_nxn.t()  # !
            normal_mask_nxn[floor_mask_nxn] = (normal_diff_nxn[floor_mask_nxn] > fc_normal_cos_thresh)
            # check update dist mask
            dist_mask_nxn[floor_mask_pad] = 0
            dist_mask_nxn = dist_mask_nxn & dist_mask_nxn.t()
            dist_mask_nxn[floor_mask_nxn] = (proj_dist_diff_nxn[floor_mask_nxn] < fc_dist_thresh)
            # TODO: check update adj mask
            adj_mask_nxn[floor_mask_pad] = 0  # to check
            adj_mask_nxn= adj_mask_nxn & adj_mask_nxn.t()
            # adj_mask_nxn[floor_mask_nxn] = 1
            adj_mask_nxn[floor_mask_nxn] = adj_mask_nxn_fc[floor_mask_nxn]

            if ceiling_height > 0:
                assert ceiling_height > floor_height
                # get ceiling mask coarse
                logger.warning("We assume that the ceiling is horizontal!")
                ceiling_normal = torch.tensor([0,0,-1]).to(device)
                ceiling_normal_diff = (mean_normal * ceiling_normal.reshape(1, 3)).sum(dim=-1)
                ceiling_normal_mask =  ceiling_normal_diff > find_fc_normal_cos_thresh
                ## use adjusted z
                ceiling_z_mask = mean_z_fn_adjusted > floor_min_z_fn_adjusted + ceiling_height
                ceiling_mask = ceiling_normal_mask & ceiling_z_mask

                if ceiling_mask.sum() > 0:
                    # get ceiling mask fine
                    ## calculate new ceiling normal
                    ceiling_prim_weight = pts_num[ceiling_mask] / pts_num[ceiling_mask].sum()
                    ceiling_normal = F.normalize((mean_normal[ceiling_mask] * ceiling_prim_weight.reshape(-1, 1)).sum(dim=0), dim=-1)
                    ## calculate new ceiling normal mask
                    ceiling_normal_diff = (mean_normal * ceiling_normal.reshape(1, 3)).sum(dim=-1)
                    ceiling_normal_mask =  ceiling_normal_diff > find_fc_normal_cos_thresh
                    ## get pairwise ceiling mask
                    ## we use the coarse ceiling z mask
                    ceiling_mask = ceiling_normal_mask & ceiling_z_mask
                    ceiling_mask_nxn = (ceiling_mask[None, :] * ceiling_mask[:, None])
                    assert (floor_mask_nxn * ceiling_mask_nxn).sum() == 0

                    # update normal mask
                    ceiling_mask_pad = ceiling_mask[:, None].repeat(1, ceiling_mask.shape[0])
                    normal_mask_nxn[ceiling_mask_pad] = 0
                    normal_mask_nxn = normal_mask_nxn & normal_mask_nxn.t()  # !
                    normal_mask_nxn[ceiling_mask_nxn] = (normal_diff_nxn[ceiling_mask_nxn] > fc_normal_cos_thresh)
                    # update dist mask
                    dist_mask_nxn[ceiling_mask_pad] = 0
                    dist_mask_nxn = dist_mask_nxn & dist_mask_nxn.t()
                    dist_mask_nxn[ceiling_mask_nxn] = (proj_dist_diff_nxn[ceiling_mask_nxn] < fc_dist_thresh)
                    # TODO: check update adj mask
                    adj_mask_nxn[ceiling_mask_pad] = 0  # to check
                    adj_mask_nxn= adj_mask_nxn & adj_mask_nxn.t()
                    # adj_mask_nxn[ceiling_mask_nxn] = 1
                    adj_mask_nxn[ceiling_mask_nxn] = adj_mask_nxn_fc[ceiling_mask_nxn]

    final_mask = adj_mask_nxn & dist_mask_nxn & normal_mask_nxn
    final_mask.fill_diagonal_(True)

    merged_groups = gpu_merge_overlapped_planes(final_mask)

    # update pts_ins_id and plane_ins_id
    pts_ins_assignment_masked_updated = torch.zeros_like(pts_ins_assignment_masked)
    used_labels = torch.stack(used_labels).squeeze()
    new_ins_id = 1
    for group in merged_groups:
        group = torch.tensor(group).long()
        try:
            cur_labels = used_labels[group]
        except:
            import pdb; pdb.set_trace()
        for old_ins_id in cur_labels:
            pts_ins_assignment_masked_updated[pts_ins_assignment_masked==old_ins_id] = new_ins_id
        new_ins_id += 1
    return pts_ins_assignment_masked_updated

def find_best_assignments_for_each_group(group_id, pts_tensor, pts_normal_tensor, pts_group_assignment_tensor, pts_primitive_assignment_tensor, start_ins_id, dist_thres=0.1):
    pts_group_assignment_tensor = pts_group_assignment_tensor.clone()
    pts_primitive_assignment_tensor = pts_primitive_assignment_tensor.clone()
    pts_ins_assignment_tensor = torch.zeros_like(pts_group_assignment_tensor)
    
    pts_activate_flag = torch.zeros_like(pts_group_assignment_tensor)
    pts_activate_flag[pts_group_assignment_tensor==group_id] = 1

    activate_prim_IDs = pts_primitive_assignment_tensor[pts_group_assignment_tensor==group_id].unique()
    activate_prim_IDs = {pid.item(): True for pid in activate_prim_IDs}

    cur_ins_id = start_ins_id
    if True:
        for ite in range(len(activate_prim_IDs)):
            valid_pts_mask = pts_activate_flag > 0
            if valid_pts_mask.sum() == 0:
                break
            valid_pts = pts_tensor[valid_pts_mask]

            best_inlier_pts_mask = None
            best_inlier_num = 0
            best_prim_id = 0  # 0 means non-plane
            # find best prim id
            for prim_id, prim_flag in activate_prim_IDs.items():
                if not prim_flag:
                    continue
                prim_pts_mask = pts_primitive_assignment_tensor==prim_id
                assert pts_activate_flag[prim_pts_mask].min() > 0 
                prim_pts = pts_tensor[prim_pts_mask]
                prim_pts_normal = pts_normal_tensor[prim_pts_mask]
                prim_normal = F.normalize(prim_pts_normal.mean(dim=0), dim=-1)
                prim_offset = -torch.mean((prim_pts * prim_normal[None, :]).sum(-1), dim=0)
                # prim_offset = -torch.median((prim_pts * prim_normal[None, :]).sum(-1), dim=0)[0]

                pts2prim_dist = (valid_pts.reshape(-1, 3) * prim_normal.reshape(1, 3)).sum(-1) + prim_offset.reshape(1)
                pts2prim_dist = pts2prim_dist.abs()
                inlier_pts_mask = pts2prim_dist < dist_thres
                inlier_num = inlier_pts_mask.sum()
                if inlier_num > best_inlier_num:
                    best_inlier_num = inlier_num
                    best_inlier_pts_mask = inlier_pts_mask
                    best_prim_id = prim_id
            
            # prim assignment
            if best_prim_id > 0:
                best_inlier_primIDs = (pts_primitive_assignment_tensor[valid_pts_mask][best_inlier_pts_mask]).unique()
                for pid in best_inlier_primIDs:
                    pts_ins_assignment_tensor[pts_primitive_assignment_tensor==pid] = cur_ins_id
                    activate_prim_IDs[pid.item()] = False
                    pts_activate_flag[pts_primitive_assignment_tensor==pid] = 0
                cur_ins_id += 1
    return pts_ins_assignment_tensor

def build_planar_mesh_for_eval(pts, pts_normal, pts_ins_assignment, faces, move_pts_on=True, return_pts=False, colorMap_vis=None):
    # ------------------------------------------------------------------- build planar mesh
    pts_ins_assignment_np = pts_ins_assignment.cpu().numpy()
    faces = faces.cpu().numpy()
    color_vis = random_color(np.unique(pts_ins_assignment_np).max().item()+100)
    if colorMap_vis is None:
        colorMap_vis = color_vis(np.unique(pts_ins_assignment_np).max().item()+10)
    pts_color = colorMap_vis[pts_ins_assignment_np] / 255.
    triangle_mesh = o3d.geometry.TriangleMesh()
    triangle_mesh.vertices = o3d.utility.Vector3dVector(pts.cpu().numpy())
    triangle_mesh.vertex_colors = o3d.utility.Vector3dVector(pts_color)
    triangle_mesh.triangles = o3d.utility.Vector3iVector(faces)
    # -------------------------------------------------------------------- label non-plane faces
    face_ass = pts_ins_assignment_np[faces.reshape(-1)].reshape(-1, 3)
    face_invalid_mask = (face_ass[:,0] == 0).astype(np.int64) + (face_ass[:,1] == 0).astype(np.int64) + (face_ass[:,2] == 0).astype(np.int64)
    # face_invalid_mask = face_invalid_mask > 0
    face_invalid_mask = face_invalid_mask == 3
    face_id = np.arange(faces.shape[0])
    removed_face_id = face_id[face_invalid_mask]
    # -------------------------------------------------------------------- update pts color
    face_max_ass = face_ass.max(axis=-1).reshape(-1, 1)
    face_max_ass = np.repeat(face_max_ass, 3, axis=-1)
    pts_ins_assignment_np[faces[~face_invalid_mask]] = face_max_ass[~face_invalid_mask]
    color_vis = random_color(np.unique(pts_ins_assignment_np).max().item()+100)
    colorMap_vis = color_vis(np.unique(pts_ins_assignment_np).max().item()+10)
    pts_color = colorMap_vis[pts_ins_assignment_np] / 255.
    triangle_mesh.vertex_colors = o3d.utility.Vector3dVector(pts_color)    
    # # -------------------------------------------------------------------- move points
    if move_pts_on:
        pts_ins_assignment_tensor = torch.from_numpy(pts_ins_assignment_np).cuda()
        pts_plane = move_pts_onto_plane_simple(pts.clone(), pts_normal, pts_ins_assignment_tensor)
        triangle_mesh.vertices = o3d.utility.Vector3dVector(pts_plane.cpu().numpy())
    else:
        pts_plane = pts
        logger.warning("using original points....")
    # -------------------------------------------------------------------- remove non-plane faces
    # logger.warning("showing all faces for debug")
    triangle_mesh.remove_triangles_by_index(removed_face_id)
    triangle_mesh.remove_duplicated_triangles()
    triangle_mesh.remove_duplicated_vertices()
    triangle_mesh.remove_unreferenced_vertices()

    # import pdb; pdb.set_trace()
    if return_pts:
        return triangle_mesh, pts_plane
    else:
        return triangle_mesh

class random_color(object):
    def __init__(self, color_num=5000):
        num_of_colors=color_num
        self.colors = ["#"+''.join([random.choice('0123456789ABCDEF') for i in range(6)])
             for j in range(num_of_colors)]

    def __call__(self, ret_n = 10):
        assert len(self.colors) > ret_n
        ret_color = np.zeros([ret_n, 3])
        for i in range(ret_n):
            hex_color = self.colors[i][1:]
            ret_color[i] = np.array([int(hex_color[j:j + 2], 16) for j in (0, 2, 4)])
        ret_color[0] *= 0
        return ret_color
        
def get_planeInsId_from_ptsInsAssignment(num_plane, pts_ins_assignment_original, pts_ins_assignment_updated):
    plane_ins_id = torch.zeros(num_plane).cuda()
    new_ids = pts_ins_assignment_updated.unique()
    for nid in new_ids:
        if nid == 0:
            continue
        pts_mask = pts_ins_assignment_updated == nid
        assert pts_mask.sum() > 0
        prim_ids = pts_ins_assignment_original[pts_mask].unique() - 1
        plane_ins_id[prim_ids] = nid.cpu().item()
    return plane_ins_id

def calculate_adj_mask(points_list, adj_ratio_threshold=0.05, adj_count_threshold=10, voxel_size=0.1):
    all_points = torch.cat(points_list)
    device = all_points.device
    plane_ids = torch.cat([torch.full((pts.size(0),), i, device=device, dtype=torch.long) 
                        for i, pts in enumerate(points_list)])
    
    base_voxels = torch.floor(all_points / voxel_size).long()
    offsets = torch.tensor(list(itertools.product([-1, 0, 1], repeat=3)), 
                        device=device, dtype=torch.long)
    expanded_voxels = (base_voxels.unsqueeze(1) + offsets).view(-1, 3)
    expanded_plane_ids = plane_ids.repeat_interleave(len(offsets))
    
    combined = torch.cat([expanded_voxels, expanded_plane_ids.unsqueeze(1)], dim=1)
    unique_combined, _ = torch.unique(combined, dim=0, return_inverse=True)
    unique_voxels = unique_combined[:, :3]
    unique_planes = unique_combined[:, 3]
    
    unique_voxel_list, inverse_idx = torch.unique(unique_voxels, dim=0, return_inverse=True)
    
    num_voxels = unique_voxel_list.size(0)
    num_planes = len(points_list)
    indices = torch.stack([inverse_idx, unique_planes], dim=0)
    voxel_plane_matrix = torch.sparse_coo_tensor(
        indices,
        torch.ones(indices.size(1), device=device, dtype=torch.float32),
        (num_voxels, num_planes)
    ).coalesce()
    
    dense_matrix = voxel_plane_matrix.to_dense()
    
    plane_voxel_counts = dense_matrix.sum(dim=0)  
    
    adj_count_matrix = torch.mm(dense_matrix.t(), dense_matrix) 
    
    ratio_AB = adj_count_matrix / (plane_voxel_counts.unsqueeze(1) + 1e-8)
    ratio_BA = adj_count_matrix / (plane_voxel_counts.unsqueeze(0) + 1e-8) 
    max_ratio = torch.maximum(ratio_AB, ratio_BA)
    
    adj_matrix = (max_ratio >= adj_ratio_threshold) & (adj_count_matrix >= adj_count_threshold)
    adj_matrix.fill_diagonal_(True)
    
    return adj_matrix.to(torch.bool)

def check_cc(
    pts, # S, 3
    pts_normal, # S, 3
    pts_ins_assignment_masked, # S
    find_fc_normal_angle_thresh=25,
    floor_height=0.2,
    ceiling_height=2.0,
    adj_ratio_threshold=0.05, 
    adj_count_threshold=10,
    voxel_size=0.1,
    fc_adj_ratio_threshold=0.0, 
    fc_adj_count_threshold=1,
    fc_voxel_size=0.1
):
    find_fc_normal_cos_thresh = math.cos(find_fc_normal_angle_thresh/180.*np.pi)

    unique_labels = pts_ins_assignment_masked.unique()

    ## all labels should be coutinuous and start from 0 or 1
    if unique_labels[0] == 0:
        assert unique_labels[-1] == len(unique_labels)-1
    else:
        assert unique_labels[-1] == len(unique_labels)

    ## get number of valid labels
    max_label_num = len(unique_labels)
    if unique_labels[0] == 0:
        max_label_num -= 1
    
    mean_normal = torch.zeros((max_label_num, 3), device="cuda")
    mean_z = torch.zeros((max_label_num, 1), device="cuda")
    used_labels = []
    pts_list = []
    pts_num = []

    count = 0
    # for label in tqdm(unique_labels, total=len(unique_labels)):
    for label in unique_labels:
        if label == 0:
            continue
        mask = pts_ins_assignment_masked == label
        assert mask.sum() > 0
        used_labels.append(label)
        # compute plane parameters and move all points onto the plane
        plane_pts = pts[mask]

        plane_normal = torch.median(pts_normal[mask], dim=0)[0]
        # plane_normal = torch.mean(original_normals_array[mask], dim=0)

        normal = plane_normal / (torch.norm(plane_normal) + 1e-10)

        z = torch.mean(plane_pts, dim=0)[-1]

        mean_normal[count] = normal
        mean_z[count] = z

        pts_list.append(plane_pts.clone())
        pts_num.append(mask.sum())

        count += 1
    
    assert len(used_labels) == count
    if count == 1:
        pts_ins_assignment_masked_updated = torch.zeros_like(pts_ins_assignment_masked)
        pts_ins_assignment_masked_updated[pts_ins_assignment_masked==used_labels[0]] = 1
        return pts_ins_assignment_masked_updated

    pts_num = torch.stack(pts_num).squeeze()

    # calculate overlap mask
    adj_mask_nxn = calculate_adj_mask(pts_list, adj_ratio_threshold, adj_count_threshold, voxel_size)
    adj_mask_nxn = adj_mask_nxn | adj_mask_nxn.t()
    adj_mask_nxn_fc = calculate_adj_mask(pts_list, fc_adj_ratio_threshold, fc_adj_count_threshold, fc_voxel_size)
    adj_mask_nxn_fc = adj_mask_nxn_fc | adj_mask_nxn_fc.t()
    
    mean_z = mean_z.squeeze()
    vec_z = torch.tensor([0,0,1]).cuda().reshape(-1, 3) * mean_z.reshape(-1, 1)  # n, 3

    if floor_height > 0:
        # get floor mask coarse
        floor_normal = torch.tensor([0,0,1]).cuda()
        floor_normal_diff = (mean_normal * floor_normal.reshape(1, 3)).sum(dim=-1)
        floor_normal_mask =  floor_normal_diff > find_fc_normal_cos_thresh

        if floor_normal_mask.sum() < 10:
            floor_mask = torch.zeros_like(floor_normal_mask)
        else:
            min_z = mean_z[floor_normal_mask].min().cpu().item()
            for i in range(10):
                floor_z_mask = mean_z < min_z + floor_height + 0.01 * i
                floor_mask = floor_normal_mask & floor_z_mask
                if floor_mask.sum() > 10:
                    break

        if floor_mask.sum() > 10:
            floor_min_z = mean_z[floor_mask].min().cpu().item()
            logger.info("floor height = %d"%(floor_min_z))
            # floor_min_z = mean_z[floor_mask].mean().cpu().item()

            # get floor mask fine
            ## calculate new floor normal
            floor_prim_weight = pts_num[floor_mask] / pts_num[floor_mask].sum()
            floor_normal = F.normalize((mean_normal[floor_mask] * floor_prim_weight.reshape(-1, 1)).sum(dim=0), dim=-1)
            ## calculate new floor normal mask
            floor_normal_diff = (mean_normal * floor_normal.reshape(1, 3)).sum(dim=-1)
            floor_normal_mask =  floor_normal_diff > find_fc_normal_cos_thresh
            ## we should align z to new floor normal !!!
            mean_z_fn_adjusted = (vec_z * floor_normal.reshape(1, 3)).sum(dim=-1)
            floor_min_z_fn_adjusted = (torch.tensor([0,0,floor_min_z]).cuda().reshape(1, 3) * floor_normal.reshape(1, 3)).sum(dim=-1).cpu().item()
            ## calculate new floor z mask
            # floor_z_mask = mean_z < floor_min_z + floor_height
            floor_z_mask = mean_z_fn_adjusted < floor_min_z_fn_adjusted + floor_height
            ## get pairwise floor mask
            floor_mask = floor_normal_mask & floor_z_mask
            floor_mask_nxn = (floor_mask[None, :] * floor_mask[:, None])

            # update adj mask
            floor_mask_pad = floor_mask[:, None].repeat(1, floor_mask.shape[0])
            adj_mask_nxn[floor_mask_pad] = 0  # to check
            adj_mask_nxn= adj_mask_nxn & adj_mask_nxn.t()
            # adj_mask_nxn[floor_mask_nxn] = 1
            adj_mask_nxn[floor_mask_nxn] = adj_mask_nxn_fc[floor_mask_nxn]

            if ceiling_height > 0:
                assert ceiling_height > floor_height
                # get ceiling mask coarse
                logger.warning("We assume that the ceiling is horizontal!")
                ceiling_normal = torch.tensor([0,0,-1]).cuda()
                ceiling_normal_diff = (mean_normal * ceiling_normal.reshape(1, 3)).sum(dim=-1)
                ceiling_normal_mask =  ceiling_normal_diff > find_fc_normal_cos_thresh
                ## use adjusted z
                ceiling_z_mask = mean_z_fn_adjusted > floor_min_z_fn_adjusted + ceiling_height
                ceiling_mask = ceiling_normal_mask & ceiling_z_mask

                logger.info("ceiling height = %d"%(floor_min_z_fn_adjusted + ceiling_height))

                if ceiling_mask.sum() > 0:
                    # get ceiling mask fine
                    ## calculate new ceiling normal
                    ceiling_prim_weight = pts_num[ceiling_mask] / pts_num[ceiling_mask].sum()
                    ceiling_normal = F.normalize((mean_normal[ceiling_mask] * ceiling_prim_weight.reshape(-1, 1)).sum(dim=0), dim=-1)
                    ## calculate new ceiling normal mask
                    ceiling_normal_diff = (mean_normal * ceiling_normal.reshape(1, 3)).sum(dim=-1)
                    ceiling_normal_mask =  ceiling_normal_diff > find_fc_normal_cos_thresh
                    ## get pairwise ceiling mask
                    ## we use the coarse ceiling z mask
                    ceiling_mask = ceiling_normal_mask & ceiling_z_mask
                    ceiling_mask_nxn = (ceiling_mask[None, :] * ceiling_mask[:, None])
                    assert (floor_mask_nxn * ceiling_mask_nxn).sum() == 0

                    # update adj mask
                    ceiling_mask_pad = ceiling_mask[:, None].repeat(1, ceiling_mask.shape[0])
                    adj_mask_nxn[ceiling_mask_pad] = 0  # to check
                    adj_mask_nxn= adj_mask_nxn & adj_mask_nxn.t()
                    # adj_mask_nxn[ceiling_mask_nxn] = 1
                    adj_mask_nxn[ceiling_mask_nxn] = adj_mask_nxn_fc[ceiling_mask_nxn]

    final_mask = adj_mask_nxn
    final_mask.fill_diagonal_(True)

    merged_groups = gpu_merge_overlapped_planes(final_mask)

    # update pts_ins_id and plane_ins_id
    pts_ins_assignment_masked_updated = torch.zeros_like(pts_ins_assignment_masked)
    used_labels = torch.stack(used_labels).squeeze()
    new_ins_id = 1
    for group in merged_groups:
        group = torch.tensor(group).long()
        try:
            cur_labels = used_labels[group]
        except:
            import pdb; pdb.set_trace()
        for old_ins_id in cur_labels:
            pts_ins_assignment_masked_updated[pts_ins_assignment_masked==old_ins_id] = new_ins_id
        new_ins_id += 1
    return pts_ins_assignment_masked_updated

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

def move_pts_onto_plane_simple(ori_pts, pts_normal, per_point_plane_assignment):
    '''
    per_point_plane_assignment: 0 means non-plane / invalid plane
    '''
    dis_list = []
    max_label = per_point_plane_assignment.max() + 1
    move_dists = []
    for label in tqdm(range(max_label), desc='moving points to plane instance...'):
        if label > 0:
            mask = per_point_plane_assignment == label
            if mask.sum() == 0:
                continue
            # compute plane parameters and move all points onto the plane
            sample_pts = ori_pts[mask].clone()
            plane_normal = torch.median(pts_normal[mask], dim=0)[0]

            plane_offset = -torch.median((sample_pts @ plane_normal))

            dist = (sample_pts @ plane_normal) + plane_offset
            projected_points = sample_pts - plane_normal * dist[:, None]

            ori_pts[mask] = projected_points

            move_dists.append(dist)
    
    # import pdb; pdb.set_trace()
    move_dists = torch.cat(move_dists, dim=0)
    avg_mv = torch.mean(move_dists)
    print(f"avg moving distance = {avg_mv}")
    print(f"max moving distance = {move_dists.abs().max()}")
    print(f"moving distance > 0.2: {(move_dists.abs() > 0.2).sum() / move_dists.shape[0] * 100}%")
    return ori_pts

def group_plane_via_normal(
    pts_normal, # S, 3
    pts_ins_assignment_masked, # S
    normal_angle_thresh=25,
    use_mean=True
):  
    normal_cos_thresh = math.cos(normal_angle_thresh/180.*np.pi)
    unique_labels = pts_ins_assignment_masked.unique()

    max_label_num = len(unique_labels)
    if unique_labels[0] == 0:
        max_label_num -= 1
    mean_normal = torch.zeros((max_label_num, 3), device="cuda")
    used_labels = []

    count = 0
    # for label in tqdm(unique_labels, total=len(unique_labels)):
    for label in unique_labels:
        if label == 0:
            continue
        mask = pts_ins_assignment_masked == label
        assert mask.sum() > 0
        used_labels.append(label)
        if use_mean:
            plane_normal = torch.mean(pts_normal[mask], dim=0)
        else:
            plane_normal = torch.median(pts_normal[mask], dim=0)[0]
        plane_normal = torch.mean(pts_normal[mask], dim=0)
        normal = plane_normal / (torch.norm(plane_normal) + 1e-10)
        mean_normal[count] = normal
        count += 1
    
    # calculate normal diff and dist diff
    normal_diff_nxn = (mean_normal[None, :] * mean_normal[:, None]).sum(2)

    # calculate normal and dist mask
    normal_mask_nxn = ((normal_diff_nxn > normal_cos_thresh))
    normal_mask_nxn = normal_mask_nxn | normal_mask_nxn.t()

    device = normal_diff_nxn.device

    final_mask = normal_mask_nxn
    final_mask.fill_diagonal_(False)

    all_idx = torch.arange(max_label_num).to(device)
    pts_ins_assignment_masked_tmp = pts_ins_assignment_masked.clone()

    for idx in range(max_label_num):
        label = used_labels[idx]
        if label == 0:
            continue
        mask = pts_ins_assignment_masked_tmp == label
        if mask.sum() == 0:
            continue
        inlier_idx = all_idx[final_mask[idx]]
        if len(inlier_idx) > 0:
            inlier_idx = inlier_idx.cpu().numpy().tolist()
            for cur_idx in inlier_idx:
                cur_label = used_labels[cur_idx]
                pts_ins_assignment_masked_tmp[pts_ins_assignment_masked_tmp==cur_label] = label

    return pts_ins_assignment_masked_tmp

def update_pts_normal(pts_plane_assignment_tensor, pts_normal_tensor, use_median=False):
    pts_normal_tensor = pts_normal_tensor.clone()
    #  ----------------------------------- update pts' normal
    max_label = pts_plane_assignment_tensor.max() + 1
    largest_ass = -1
    largest_size = 0
    for label in range(max_label):
        if label > 0:
            mask = pts_plane_assignment_tensor == label
            if mask.sum() == 0:
                continue
            if mask.sum() > largest_size:
                largest_size = mask.sum()
                largest_ass = label
            if use_median:
                normal_cur = torch.median(pts_normal_tensor[mask], dim=0)[0]
            else:
                normal_cur = torch.mean(pts_normal_tensor[mask], dim=0)
            try:
                pts_normal_tensor[mask] = normal_cur.reshape(1, 3).repeat(mask.sum(), 1)
            except:
                import pdb; pdb.set_trace()
    return pts_normal_tensor

def get_continues_pts_ins_assignment(pts_plane_assignment):
    labels = pts_plane_assignment.unique()
    pts_plane_assignment_new = torch.zeros_like(pts_plane_assignment)
    new_label = 1
    for label in labels:
        if label > 0:
            pts_plane_assignment_new[pts_plane_assignment == label] = new_label
            new_label += 1
    # assert pts_plane_assignment_new.min() > 0
    return pts_plane_assignment_new

def calculate_pts2mesh_dist(pts, coarse_mesh_o3d):
    mesh_vertices = torch.from_numpy(np.array(coarse_mesh_o3d.vertices)).to(pts.device).float()
    
    pts_batch = pts.unsqueeze(0)  # (1, N, 3)
    mesh_batch = mesh_vertices.unsqueeze(0)  # (1, M, 3)
    
    dists, idx, _ = knn_points(pts_batch, mesh_batch, K=1)
    
    pts_dist2mesh_tensor = torch.sqrt(dists.squeeze(-1).squeeze(0))
    
    return pts_dist2mesh_tensor

def sample_pts_from_GivenPlanePrim(
        plane_normal, 
        plane_center, 
        plane_radii, 
        plane_rot_q, 
        scene_scale=1.0,
        scene_offset=[0., 0., 0.],
        space_resolution=0.05, 
        plane_ins_id=None
        ):
    """
    space_resolution=0.05 --> 0.05 meter
    """
    sample_interval = space_resolution * scene_scale  # to normalized space

    rot_q = F.normalize(plane_rot_q, dim=-1)  # n, 4
    rot_matrix = quat_to_rot(rot_q)  # n, 3, 3
    plane_normal_standard = torch.zeros_like(plane_normal)
    plane_normal_standard[..., -1] = 1
    radii_x = plane_radii[..., 0]  # n
    radii_y = plane_radii[..., 1]  # n

    if plane_ins_id is None:
        plane_ins_id = torch.arange(plane_center.shape[0]) + 1 # valid plane ins IDs start from 1
    plane_ins_id = plane_ins_id.reshape(-1).int()  # 0 means non-plane ins
            
    pts = []
    per_point_plane_assignment = []
    normals = []
    faces = None

    for i in tqdm(range(len(plane_normal)), desc='sampleing points from planes...'):
        rx, ry = radii_x[i].item(), radii_y[i].item()
        nx, ny = int(max(2 * rx // sample_interval + 1, 3)), int(max(2 * ry // sample_interval + 1, 3))

        px = torch.linspace(-rx, rx, nx).reshape(-1, 1, 1).repeat(1, ny, 1).cuda()
        py = torch.linspace(-ry, ry, ny).reshape(1, -1, 1).repeat(nx, 1, 1).cuda()
        pz = torch.zeros_like(py)
        pts_i = torch.cat([px, py, pz], dim=-1).reshape(-1, 3)
        pts_i_transformed = torch.mm(rot_matrix[i], pts_i.permute(1, 0)).permute(1, 0) + plane_center[i].reshape(1, 3)

        normal_i = plane_normal[i].reshape(1, 3).repeat(pts_i_transformed.shape[0], 1)
        normals.append(normal_i)

        pts.append(pts_i_transformed)

        pts_planeID = plane_ins_id[i] * torch.ones(pts_i_transformed.shape[0]).to(pts_i_transformed.device)
        per_point_plane_assignment.append(pts_planeID.int())

        faces_i = []
        for i in range(nx*ny - ny):
            if (i+1) % (ny) == 0:
                continue
            faces_i.append([i, i+ny, i+1])
            faces_i.append([i+1, i+ny, i+ny+1])
        faces_i = torch.from_numpy(np.array(faces_i)).cuda()
        
        if faces is None:
            faces = faces_i
            cur_pts_num = nx*ny
        else:
            faces_i += cur_pts_num
            faces = torch.cat([faces, faces_i], dim=0)
            cur_pts_num += nx*ny

    pts = torch.cat(pts, dim=0).detach()
    pts /= scene_scale
    scene_offset = torch.tensor(scene_offset).cuda().reshape(1, 3)
    pts += scene_offset

    faces = faces.detach()
    normals = torch.cat(normals, dim=0).detach()

    per_point_plane_assignment = torch.cat(per_point_plane_assignment, dim=0).detach()

    return pts, per_point_plane_assignment, normals, faces


def plot_rectangle_planes(plane_centers, plane_normals, plane_radii, rot_q, epoch=-1, suffix='', to_unscaled_coord=True, pose_cfg=None, out_path=None, plane_id=None, color_type=''):
    plane_normals_standard = torch.zeros_like(plane_normals)
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
    rot_q = F.normalize(rot_q, dim=-1)  # n, 4
    rot_matrix = quat_to_rot(rot_q)  # n, 3, 3
    vertices_transformed = torch.bmm(rot_matrix, vertices_standard.permute(0, 2, 1)).permute(0, 2, 1) + plane_centers[:, None]  # n, 4, 3
    vertices_all = vertices_transformed.reshape(-1, 3).detach().cpu().numpy()  # 4n, 3
    
    if to_unscaled_coord:
        assert pose_cfg is not None
        vertices_all /= pose_cfg.scale
        vertices_all += pose_cfg.offset
    else:
        suffix = suffix + '_normalized'

    N = vertices_all.shape[0] // 4
    triangle_mesh = o3d.geometry.TriangleMesh()
    triangle_mesh.vertices = o3d.utility.Vector3dVector(vertices_all)

    if color_type == 'normal':
        normal_color = (plane_normals + 1.) / 2.
        normal_color = normal_color.reshape(-1, 1, 3).repeat(1, 4, 1)
        colors = normal_color.detach().cpu().numpy().reshape(-1, 3)
        triangle_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        suffix = suffix + '_colorNormal'
    elif color_type == 'prim':
        if plane_id is None:
            plane_id = torch.arange(plane_centers.shape[0])
        plane_id = plane_id.reshape(-1, 1).repeat(1, 4).int().cuda()
        color_vis = random_color()
        colorMap_vis = color_vis(plane_id.unique().max().item() + 1)
        colors = colorMap_vis[plane_id.detach().cpu().numpy().reshape(-1)]
        colors = colors.astype(np.float64) / 255.
        triangle_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        suffix = suffix + '_colorPrim'
    else:
        pass

    idx_v1 = np.arange(N).reshape(-1, 1) * 4
    idx_v2 = idx_v1 + 1
    idx_v3 = idx_v1 + 2
    idx_v4 = idx_v1 + 3

    triangle_vIdx1 = np.concatenate([idx_v1, idx_v2, idx_v3], axis=-1)  # 2n, 3
    triangle_vIdx2 = np.concatenate([idx_v3, idx_v4, idx_v1], axis=-1)  # 2n, 3
    triangle_vIdx = np.concatenate([triangle_vIdx1, triangle_vIdx2], axis=0)  # 4n, 3
    triangle_mesh.triangles = o3d.utility.Vector3iVector(triangle_vIdx)

    if os.path.exists(out_path):
        if epoch == -1:
            os.makedirs(out_path, exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(out_path, "planarSplat_%s.ply" % (suffix)), triangle_mesh)
        else:
            str_epoch = ('%d' % (epoch)).zfill(3)
            os.makedirs(out_path, exist_ok=True)
            o3d.io.write_triangle_mesh(os.path.join(out_path, "planarSplat_%s_%s.ply" % (str_epoch, suffix)), triangle_mesh)
    return triangle_mesh


def refuse_mesh(depths: List[np.ndarray], 
                poses: List[np.ndarray], 
                intrinsics: List[np.ndarray], 
                H: int, 
                W: int, 
                voxel_length: float=0.05, 
                sdf_trunc: float=0.08, 
                depth_trunc: float=5.0
                ):
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
    )
    
    for pose, K, depth_pred in tqdm(zip(poses, intrinsics, depths)):
        depth_pred = depth_pred.reshape(H, W)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = K
        
        rgb = np.ones((H, W, 3))
        rgb = (rgb * 255).astype(np.uint8)
        rgb = o3d.geometry.Image(rgb)
        
        depth_pred = o3d.geometry.Image(depth_pred)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb, depth_pred, depth_scale=1.0, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
        )
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
        extrinsic = np.linalg.inv(pose)
        volume.integrate(rgbd, intrinsic, extrinsic)
    
    return volume.extract_triangle_mesh()

def get_coarse_mesh(poses: List,
                    intrinsics: List,
                    depths: List, 
                    H: int, 
                    W: int, 
                    voxel_length: float=0.02, 
                    sdf_trunc: float=0.08
                    ):
    
    poses = [x for x in poses]
    intrinsics = [x for x in intrinsics]
    depths = [x.cpu().numpy().reshape(H, W) for x in depths]

    mesh = refuse_mesh(depths, poses, intrinsics, H, W, voxel_length=voxel_length, sdf_trunc=sdf_trunc)
    return mesh

def get_rast_seg_map(rgb_insIDs, plane_insIDs, normal_map_local=None, offset_map_local=None, min_mask_size=200):
    # get instance segmentation map and segmentation masks
    max_id = plane_insIDs.max()
    rgb_insIDs = rgb_insIDs * max_id
    valid_mask = (rgb_insIDs[0:1] == rgb_insIDs[1:2]) & (rgb_insIDs[0:1] == rgb_insIDs[2:3])
    rgb_insIDs = rgb_insIDs * valid_mask
    seg_masks = []
    seg_map_global = torch.zeros_like(rgb_insIDs[0])
    seg_map_local = torch.zeros_like(rgb_insIDs[0])
    seg_params = []
    local_id = 1
    for id in plane_insIDs.unique():
        if id == 0:  # ignore the non-planar area, e.g. the pixle's value is 0
            continue
        mask_cur = (rgb_insIDs[0] >= (id - 0.01)) & (rgb_insIDs[0] <= (id + 0.01))
        if mask_cur.sum() <= min_mask_size:
            continue
        seg_masks.append(mask_cur)
        seg_map_global[mask_cur] = id

        seg_map_local[mask_cur] = local_id
        local_id += 1

        if offset_map_local is not None and normal_map_local is not None:
            cur_n = torch.median(normal_map_local[mask_cur.reshape(-1)],dim=0)[0]
            cur_o = torch.median(offset_map_local[mask_cur.reshape(-1)],dim=0)[0]
            seg_params.append(cur_n * cur_o)
    
    return seg_map_global, seg_map_local, seg_params, seg_masks

if __name__ == '__main__':
    plane_num = 1000
    plane_center = torch.randn(plane_num, 3).cuda() * 10. # n, 3
    plane_radii = torch.rand(plane_num, 2).cuda() * 0.5  # n, 2
    plane_rot_q = F.normalize(torch.randn(plane_num, 4), dim=-1).cuda()  # n, 4

    plane_rot_matrix = quat_to_rot(plane_rot_q)
    plane_normal_standard = torch.tensor([0., 0., 1.]).reshape(1, 3).cuda()
    plane_normal = torch.bmm(plane_rot_matrix, plane_normal_standard.reshape(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)

    merge_plane(
        plane_normal, # n, 3
        plane_center, # n, 3
        plane_radii,  # n, 2
        plane_rot_q,  # n, 4
        )