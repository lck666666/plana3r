import torch
import torch.nn.functional as F
import numpy as np
import math
import itertools
from planar_splatting.utils.model_util import quat_to_rot, get_rotation_quaternion_of_normal, quaternion_mult

GRID_N = 7
xs = torch.linspace(-1, 1, GRID_N)
ys = torch.linspace(-1, 1, GRID_N)
GRID_X, GRID_Y = torch.meshgrid(ys, xs, indexing='ij')  # 注意indexing参数
GRID_X = GRID_X.cuda()
GRID_Y = GRID_Y.cuda()
GRID_XY = torch.stack((GRID_X, GRID_Y), dim=-1).cuda()

def convert_groups_to_insIDs(groups, prim_num):
    prim_num_ = sum([len(gr) for gr in groups])
    assert prim_num == prim_num_
    insIDs = torch.full((prim_num,), fill_value=-1, dtype=torch.long).cuda()
    for g_idx in range(len(groups)):
        cur_group = torch.tensor(groups[g_idx]).cuda()
        insIDs[cur_group] = g_idx
    return insIDs

def mask_group(input_mask, confidence=None):
    final_mask = input_mask.clone()
    final_mask.fill_diagonal_(False)

    device = final_mask.device

    max_label_num = final_mask.shape[0]
    all_idx = torch.arange(max_label_num).to(device)
    state = torch.ones_like(all_idx)

    if confidence is not None:
        sorted_confidence, indices = torch.sort(confidence, descending=True)
    else:
        indices = all_idx

    groups = []
    for idx in indices:
        if state[idx] == 0:
            continue
        inlier_idx = all_idx[final_mask[idx]]
        if len(inlier_idx) > 0:
            group_ = inlier_idx.cpu().numpy().tolist()
            group = [idx.item()] + group_

            state[inlier_idx] = 0
            state[idx] = 0

            final_mask[inlier_idx] = False
            final_mask[idx] = False

            final_mask[:,inlier_idx] = False
            final_mask[:,idx] = False
        else:
            group = [idx.item()]
            state[idx] = 0
            final_mask[idx] = False
            final_mask[:,idx] = False

        groups.append(group)
    
    return groups

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

def check_cc(adj_mask):
    adj_mask.fill_diagonal_(True)
    merged_groups = gpu_merge_overlapped_planes(adj_mask)
    return merged_groups

def get_normal_mask(normals, normal_angle_thresh=10):
    normal_cos_thresh = math.cos(normal_angle_thresh/180.*np.pi)

    # calculate normal diff and dist diff
    normal_diff_nxn = (normals[None, :] * normals[:, None]).sum(2)
    # calculate normal mask
    normal_mask_nxn = normal_diff_nxn > normal_cos_thresh

    return normal_mask_nxn

def compute_offset(pts, normals):
    v1 = pts
    v2 = F.normalize(normals, dim=-1)
    offset = (v1 * v2).sum(-1)
    return offset

def get_p2p_dist_mask(plane_normal, plane_center, dist_thresh=0.1):
    plane_offset = compute_offset(plane_center, plane_normal)

    pts = plane_center

    planes_normal = plane_normal.reshape(1, -1, 3)  # 1, N, 3
    planes_offset = plane_offset.reshape(1, -1, 1)
    pts = pts.reshape(-1, 1, 3)

    pts_origin2planes = planes_normal * planes_offset  # 1, N, 3
    dist_field = ((pts - pts_origin2planes) * planes_normal).sum(-1, keepdim=True)  # N, N, 1

    mask = dist_field.abs() < dist_thresh
    mask = mask.squeeze()

    mask = mask & mask.t()

    return mask

def calculate_adj_mask(points_list, adj_ratio_threshold=0.05, adj_count_threshold=10, voxel_size=0.1):
    all_points = torch.cat(points_list, dim=0)
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

def sample_pts_from_PlanePrim(plane_center, plane_radii, plane_xAxis, plane_yAxis, return_list=False):
    plane_num = plane_center.shape[0]
    radii_x = plane_radii[..., 0]  # n
    radii_y = plane_radii[..., 1]  # n

    pts_x = GRID_X.reshape(1, -1, 1) * plane_xAxis.reshape(-1, 1, 3) * radii_x.reshape(-1, 1, 1) # n_plane, n_pts, 3
    pts_y = GRID_Y.reshape(1, -1, 1) * plane_yAxis.reshape(-1, 1, 3) * radii_y.reshape(-1, 1, 1) # n_plane, n_pts, 3

    pts = plane_center.reshape(-1, 1, 3) + pts_x + pts_y # n_plane, n_pts, 3

    if return_list:
        pts = [pts[i] for i in range(plane_num)]
        return pts
    else:
        return pts

def plane_cluster(normals, radiis, centers, rots_q, normal_angle_thresh=10, dist_thresh=0.1):
    '''
    normals: n, 3
    ''' 
    plane_xAxis_standard = torch.tensor([1., 0., 0.]).reshape(1, 3).cuda()
    plane_yAxis_standard = torch.tensor([0., 1., 0.]).reshape(1, 3).cuda()

    plane_num = normals.shape[0]

    plane_size = radiis[:,0] * radiis[:,1]
    normal_mask = get_normal_mask(normals, normal_angle_thresh=normal_angle_thresh)
    dist_mask = get_p2p_dist_mask(normals, centers, dist_thresh=dist_thresh)
    mask = normal_mask & dist_mask
    groups = mask_group(input_mask=mask, confidence=plane_size)
    insIDs = convert_groups_to_insIDs(groups, normals.shape[0])
    mask2 = torch.zeros_like(mask)
    mask2.fill_diagonal_(True)
    for gp in groups:
        if len(gp) == 1:
            continue
        mask_gp = torch.zeros_like(mask).int()
        mask_gp[gp] += 1
        mask_gp[:, gp] += 1
        mask2[mask_gp == 2] = True

    rots_q = F.normalize(rots_q, dim=-1)
    plane_rots_matrix = quat_to_rot(rots_q)  # n, 3, 3
    plane_xAxis = torch.bmm(plane_rots_matrix, plane_xAxis_standard.reshape(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)  # n, 3
    plane_yAxis = torch.bmm(plane_rots_matrix, plane_yAxis_standard.reshape(-1, 3, 1).expand(plane_num, 3, 1)).squeeze(-1)  # n, 3
    pts_list = sample_pts_from_PlanePrim(centers, radiis, plane_xAxis, plane_yAxis, return_list=True)
    adj_mask = calculate_adj_mask(pts_list, adj_ratio_threshold=0, adj_count_threshold=1, voxel_size=0.1)
    adj_mask = adj_mask & mask2
    groups = check_cc(adj_mask)
    insIDs = convert_groups_to_insIDs(groups, normals.shape[0])
    print(len(groups))

    return insIDs

def plane_adjust(plane_ins_IDs, normals, centers, rots_q):
    ins_IDs = torch.unique(plane_ins_IDs)
    normals_new = normals.clone()
    centers_new = centers.clone()
    rots_q_new = rots_q.clone()

    for id in ins_IDs:
        mask = plane_ins_IDs == id
        pts_tmp = centers[mask].view(-1, 3)  # m, 3
        normals_tmp = normals[mask].view(-1, 3)
        rots_q_tmp = rots_q[mask].view(-1, 4)

        median_normal = torch.median(normals_tmp, dim=0)[0]  # 3,
        median_normal = F.normalize(median_normal, dim=-1)
        median_offsets = -torch.median(torch.matmul(pts_tmp, median_normal.reshape(3, 1)).squeeze())
    
        dist = (pts_tmp @ median_normal.view(3, 1)) + median_offsets.view(1, 1) # npts, 1
        projected_points = pts_tmp - median_normal.view(1, 3) * dist

        normals_new[mask] = median_normal
        centers_new[mask] = projected_points

        rots_delta = get_rotation_quaternion_of_normal(median_normal.view(-1, 3), normals_tmp)
        rots_delta = F.normalize(rots_delta.cuda(), dim=-1)

        rots_q_ = quaternion_mult(rots_delta, rots_q_tmp)
        rots_q_new[mask] = rots_q_

    return normals_new, centers_new, rots_q_new



if __name__ == '__main__':
    pts = torch.rand(2048*10, 3).cuda()
    plane_num = 2048
    pts_list = [torch.rand(20, 3).cuda() for i in range(plane_num)]

    import time
    t1 = time.time()
    calculate_adj_mask(pts_list, adj_ratio_threshold=0.05, adj_count_threshold=10, voxel_size=0.1)
    print(time.time() - t1)

    