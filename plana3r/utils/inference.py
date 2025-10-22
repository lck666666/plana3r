import torch, numpy as np
import torch.nn.functional as F

from diff_rect_rasterization import RectRasterizationSettings, RectRasterizer 
from plana3r.utils.merge_tools import merge_plane as merge_plane_dev
from plana3r.utils.merge_util_for_ref import merge_plane as merge_plane_stable
from planar_splatting.utils import plot_util
from planar_splatting.utils.model_util import quat_to_rot, rot_to_quat, quaternion_mult

import open3d as o3d
import random
from PIL import Image
import quaternion
import os, cv2

def rast_primitives(view_info_list, plane_center_total, plane_radii_total, plane_rot_q_total, plane_insIDs_normlized_colors=None, plane_insIDs=None, height=384, width=512):
    rgb_insIDs_list = []
    allmap_list = []

    if isinstance(plane_center_total, list):
        plane_center_total = torch.cat(plane_center_total, dim=0).cuda()
    if isinstance(plane_radii_total, list):
        plane_radii_total = torch.cat(plane_radii_total, dim=0).cuda()
    if isinstance(plane_rot_q_total, list):
        plane_rot_q_total = torch.cat(plane_rot_q_total, dim=0).cuda()

    if plane_insIDs_normlized_colors is None:
        plane_insIDs_normlized_colors = torch.arange(plane_center_total.shape[0]).cuda().float() + 1
        plane_insIDs_normlized_colors = plane_insIDs_normlized_colors.reshape(-1, 1).repeat(1, 3)
    elif isinstance(plane_insIDs_normlized_colors, list):
        plane_insIDs_normlized_colors = torch.cat(plane_insIDs_normlized_colors, dim=0).cuda()

    if plane_insIDs is None:
        plane_insIDs = torch.arange(plane_center_total.shape[0]).cuda().float() + 1
    elif isinstance(plane_insIDs, list):
        plane_insIDs = torch.cat(plane_insIDs, dim=0).cuda()
    valid_mask = plane_insIDs > 0

    assert plane_insIDs_normlized_colors.shape[0] == plane_center_total.shape[0]
    assert plane_insIDs_normlized_colors.shape[0] == plane_insIDs.shape[0]

    plane_center_total = plane_center_total[valid_mask]
    plane_radii_total = plane_radii_total[valid_mask]
    plane_rot_q_total = plane_rot_q_total[valid_mask]
    plane_insIDs_normlized_colors = plane_insIDs_normlized_colors[valid_mask]

    for v in range(len(view_info_list)):
        splat_weight = 300
        tanfovx = view_info_list[v].tanfovx
        tanfovy = view_info_list[v].tanfovy
        raster_cam_w2c = view_info_list[v].raster_cam_w2c
        raster_cam_fullproj = view_info_list[v].raster_cam_fullproj
        raster_cam_center = view_info_list[v].raster_cam_center
        raster_img_center = view_info_list[v].raster_img_center

        screenspace_points = torch.zeros_like(plane_center_total, dtype=plane_center_total.dtype, requires_grad=True, device="cuda")
        try:
            screenspace_points.retain_grad()
        except:
            pass

        raster_settings = RectRasterizationSettings(
                            image_height=height,
                            image_width=width,
                            tanfovx=tanfovx,
                            tanfovy=tanfovy,
                            bg=torch.tensor([0., 0., 0.]).cuda(),
                            scale_modifier=1.0,
                            viewmatrix=raster_cam_w2c,      #images-pair coordinates
                            projmatrix=raster_cam_fullproj, #images-pair coordinates
                            sh_degree=0,
                            campos=raster_cam_center,
                            prefiltered=False,
                            debug=False,
                            lambdaw=splat_weight * 5.0,
                            image_center=raster_img_center,
                            scales2=plane_radii_total[:,:2].cuda(),
                            hard_render=True)
            
        # ======================================= plane model forward
        rasterizer = RectRasterizer(raster_settings=raster_settings)
        with torch.no_grad():
            rgb_insIDs, _, allmap = rasterizer(
                        means3D = plane_center_total.cuda(), #images-pair coordinates
                        means2D = screenspace_points,
                        shs = None,
                        colors_precomp = plane_insIDs_normlized_colors,
                        opacities = torch.ones_like(plane_center_total.cuda())[:, :1],
                        scales = plane_radii_total.cuda(),
                        rotations = plane_rot_q_total.cuda(),
                        cov3D_precomp = None
            )
        rgb_insIDs_list.append(rgb_insIDs)
        allmap_list.append(allmap)
    return rgb_insIDs_list, allmap_list

def get_per_view_rast_segmap_segparam(allmap, view_info_v, rgb_insIDs, plane_insIDs, min_mask_size=200):
    # get local normal map and depth map
    normal_map_local = allmap[2:5].reshape(3, -1).transpose(1, 0)  # hw, 3
    depth_map = allmap[0:1]  # 1, h, w
    intrinsic = view_info_v.intrinsic  # 3, 3
    image_height, image_width = depth_map.shape[-2], depth_map.shape[-1]
    uv = np.mgrid[0:image_height, 0:image_width].astype(np.int32)
    uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float().cuda()
    uv = uv.reshape(2, -1).transpose(1, 0)  # h*w, 2
    uv1 = torch.cat([uv, torch.ones_like(uv[:, 0:1])], dim=-1)  # hw, 3
    intrinsic_inv = torch.inverse(intrinsic)  # 3, 3
    pts3d_local = ((intrinsic_inv @ uv1.t()) * depth_map.reshape(1, -1)).transpose(1, 0)  # hw, 3
    offset_map_local = -torch.bmm(normal_map_local[:, None], pts3d_local[..., None]).squeeze()
    offset_map_local = offset_map_local.reshape(-1)  # h, w
    assert offset_map_local.min() >= 0
    normal_map_local = - normal_map_local   # Important !!

    seg_map_global, seg_map_local, seg_params, seg_masks = get_rast_seg_map(rgb_insIDs, plane_insIDs, normal_map_local, offset_map_local, min_mask_size=min_mask_size)
    
    return seg_map_global, seg_map_local, seg_params, seg_masks

def erode_mask(mask, kernel_size=3):
    """
    使用卷积模拟腐蚀操作
    mask: (B, 1, H, W) 或 (B, H, W)
    kernel_size: 结构元素大小（如 3x3）
    """
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # 添加 channel 维度

    # 定义卷积核（全 1）
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=mask.device)

    # 使用卷积统计每个位置周围有多少个 1
    pad_size = kernel_size // 2
    padded = F.pad(mask, (pad_size, pad_size, pad_size, pad_size), mode='constant', value=0)
    conv_sum = F.conv2d(padded, kernel, padding=0, stride=1)

    # 只有当所有 neighbor 都是 1 时，sum 才等于 kernel_size**2
    eroded = (conv_sum == kernel_size ** 2).float()
    return eroded

def get_rast_seg_map(rgb_insIDs, plane_insIDs, normal_map_local=None, offset_map_local=None, min_mask_size=200):
    # get instance segmentation map and segmentation masks
    max_id = plane_insIDs.max()
    device = rgb_insIDs.device
    rgb_insIDs = rgb_insIDs * max_id
    valid_mask = (rgb_insIDs[0:1] == rgb_insIDs[1:2]) & (rgb_insIDs[0:1] == rgb_insIDs[2:3])
    rgb_insIDs = rgb_insIDs * valid_mask
    seg_masks = []
    seg_map_global = torch.zeros(rgb_insIDs.shape[1:], dtype=torch.int, device=device)
    seg_map_local = seg_map_global.clone()
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

        for ks in [7, 5, 3]:
            mask_cur_erode = erode_mask(mask_cur[None].float(), kernel_size=ks).squeeze() > 0
            if mask_cur_erode.sum() > 0:
                mask_cur = mask_cur_erode
                break

        if offset_map_local is not None and normal_map_local is not None:
            cur_n = torch.median(normal_map_local[mask_cur.reshape(-1)],dim=0)[0]
            cur_o = torch.median(offset_map_local[mask_cur.reshape(-1)],dim=0)[0]
            # cur_o = torch.mean(offset_map_local[mask_cur.reshape(-1)],dim=0)
            seg_params.append(cur_n * cur_o)
    
    return seg_map_global, seg_map_local, seg_params, seg_masks

def merge_primitives(plane_normal, plane_center, plane_radii, plane_rot_q, merge_version):
    if isinstance(plane_normal, list):
        plane_normal = torch.cat(plane_normal, dim=0).cuda()
    if isinstance(plane_center, list):
        plane_center = torch.cat(plane_center, dim=0).cuda()
    if isinstance(plane_radii, list):
        plane_radii = torch.cat(plane_radii, dim=0).cuda()
    if isinstance(plane_rot_q, list):
        plane_rot_q = torch.cat(plane_rot_q, dim=0).cuda()
    
    if merge_version == 'dev':
        plane_insIDs, valid_mask = merge_plane_dev(
            plane_normal, 
            plane_center, 
            plane_radii[:,:2],  
            plane_rot_q, 
            normal_angle_thresh = 25,
            cc_dist_thresh=0.05,
            # cc_dist_thresh=0.015,
            area_thresh = 0.2**2,
            area_thresh2=0.4**2,
            cc_delta=0.05
            )
    else:
        _, plane_insIDs, _, _, _ = merge_plane_stable(
            plane_normal, 
            plane_center, 
            plane_radii[:,:2],  
            plane_rot_q,  
            coarse_mesh_o3d=None,
            return_ins_parameters=True,
            normal_angle_thresh=25,
            dist_thresh=0.1, 
            space_resolution=0.05,
            voxel_size=0.02,
        )
        plane_insIDs = plane_insIDs.int()
        valid_mask = plane_insIDs > 0
    max_id = plane_insIDs.max()

    assert max_id > 0
    return plane_insIDs, valid_mask

def draw_prims(plane_normal, plane_center, plane_radii,  plane_rot_q, suffix='rawPrims', epoch=-1, to_unscaled_coord=False, plane_id=None, plot_dir ="outputs_ply", flip_3d=False):
    if isinstance(plane_normal, list):
        plane_normal = torch.cat(plane_normal, dim=0)
    if isinstance(plane_center, list):
        plane_center = torch.cat(plane_center, dim=0)
    if isinstance(plane_radii, list):
        plane_radii = torch.cat(plane_radii, dim=0)
    if isinstance(plane_rot_q, list):
        plane_rot_q = torch.cat(plane_rot_q, dim=0)

    # normal_mesh = plot_util.plot_rectangle_planes(
    #     plane_center, plane_normal, plane_radii, plane_rot_q, 
    #     epoch=epoch, 
    #     suffix='%s'%(suffix), 
    #     to_unscaled_coord=to_unscaled_coord, 
    #     pose_cfg=None, 
    #     out_path=plot_dir,
    #     plane_id=plane_id, 
    #     color_type='normal')
    prim_mesh = plot_util.plot_rectangle_planes(
        plane_center, plane_normal, plane_radii, plane_rot_q, 
        epoch=epoch, 
        suffix='%s'%(suffix), 
        to_unscaled_coord=to_unscaled_coord, 
        pose_cfg=None, 
        out_path=plot_dir,
        plane_id=plane_id, 
        color_type='prim',
        flip_3d=flip_3d)

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

def get_random_color_map(max_ins_num=10000):
    return random_color(max_ins_num)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count - 1, -1, -1)])

def labelcolormap(N):
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r = 0
        g = 0
        b = 0
        id = i
        for j in range(7):
            str_id = uint82bin(id)
            r = r ^ (np.uint8(str_id[-1]) << (7 - j))
            g = g ^ (np.uint8(str_id[-2]) << (7 - j))
            b = b ^ (np.uint8(str_id[-3]) << (7 - j))
            id = id >> 3
        cmap[i, 0] = b
        cmap[i, 1] = g
        cmap[i, 2] = r
    return cmap

def plot_segmentation(segmentation, image, plot_dir, suffix=''):
    # print("Notice: please ensure that the non-plane idx is 0!")

    colors = labelcolormap(256)
    # ***************  get color segmentation
    seg = np.stack([colors[segmentation, 0], colors[segmentation, 1], colors[segmentation, 2]], axis=2)
    # ***************  get blend image
    blend_seg = (seg * 0.7 + image * 0.3).astype(np.uint8)
    seg_mask = (segmentation > 0).astype(np.uint8)
    seg_mask = seg_mask[:, :, np.newaxis]
    blend_seg = blend_seg * seg_mask + image.astype(np.uint8) * (1 - seg_mask)
    # ***************  save
    blend_seg_path = os.path.join(plot_dir, f'output_seg_blend_{suffix}.png')
    cv2.imwrite(blend_seg_path, blend_seg)

def get_coordinate_map(raw_K, raw_h, raw_w, oh, ow, device):
    # Calculate K_inv * xy1, taking into account that the image has been scaled (oh/ow->h/w), 
    # and if there are any other image processing steps, they should be included in the calculation.
    scale_y = oh / raw_h
    scale_x = ow / raw_w
    out_K = raw_K.copy()
    out_K[0] *= scale_x
    out_K[1] *= scale_y

    K_inv = np.linalg.inv(np.array(out_K))
    K_inv = torch.FloatTensor(K_inv).to(device)

    x = torch.arange(ow, dtype=torch.float32).view(1, ow)
    y = torch.arange(oh, dtype=torch.float32).view(oh, 1)

    x = x.to(device)
    y = y.to(device)
    xx = x.repeat(oh, 1)
    yy = y.repeat(1, ow)
    xy1 = torch.stack((xx, yy, torch.ones((oh, ow), dtype=torch.float32).to(device)))  # (3, h, w)
    xy1 = xy1.view(3, -1)  # (3, h*w)

    k_inv_dot_xy1 = torch.matmul(K_inv, xy1)  # (3, h*w)

    return k_inv_dot_xy1

def writePCDFile(folder, suffix, depth_list, segmentation_list, image_list, nonplane_idx, out_K_inv_dot_xy_1_list, out_h, out_w, c2w_seq_list=None, flip_3d=False):
    if not isinstance(depth_list, list):
        depth_list = [depth_list]
    if not isinstance(segmentation_list, list):
        segmentation_list = [segmentation_list]
    if not isinstance(image_list, list):
        image_list = [image_list]
    if not isinstance(out_K_inv_dot_xy_1_list, list):
        out_K_inv_dot_xy_1_list = [out_K_inv_dot_xy_1_list]

    pts3d_list = []
    ptsRGB_list = []
    for v, (depth, image, segmentation, out_K_inv_dot_xy_1) in enumerate(zip(depth_list, image_list, segmentation_list, out_K_inv_dot_xy_1_list)):
        depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        segmentation = cv2.resize(segmentation, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        pts3d = out_K_inv_dot_xy_1 * depth[None] # 3, h, w
        pts3d = pts3d.reshape(3, -1) # 3, hw
        if c2w_seq_list is not None:
            c2w = c2w_seq_list[v]
            pts3d_hom = np.concatenate([pts3d, np.ones((1, pts3d.shape[1]))], axis=0) # 4, hw
            pts3d_hom = np.dot(c2w, pts3d_hom) # 4, hw
            pts3d = pts3d_hom[:3] # 3, hw
        
        mask = (segmentation > nonplane_idx).reshape(-1) & (depth > 0).reshape(-1)
        pts3d_list.append(pts3d.transpose(1, 0)[mask])
        ptsRGB_list.append(image.reshape(-1, 3)[mask])
    
    # save color pcd with open3d
    pts3d = np.concatenate(pts3d_list, axis=0)
    if flip_3d:
        pts3d[:, 1] *= -1
        pts3d[:, 2] *= -1
    ptsRGB = np.concatenate(ptsRGB_list, axis=0).astype(np.float32)[:, ::-1]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts3d)
    pcd.colors = o3d.utility.Vector3dVector(ptsRGB / 255.0)
    o3d.io.write_point_cloud(folder + '/'+ f'planar_{suffix}_pcd.ply', pcd)
    return

def writePLYFile(folder, suffix, depth_list, segmentation_list, image_list, nonplane_idx, out_K_inv_dot_xy_1_list, out_h, out_w, c2w_seq_list=None, flip_3d=False):
    if not isinstance(depth_list, list):
        depth_list = [depth_list]
    if not isinstance(segmentation_list, list):
        segmentation_list = [segmentation_list]
    if not isinstance(image_list, list):
        image_list = [image_list]
    if not isinstance(out_K_inv_dot_xy_1_list, list):
        out_K_inv_dot_xy_1_list = [out_K_inv_dot_xy_1_list]
    faces_list = []
    faces_num_list = []
    depth_resiezed_list = []
    image_resized_list = []
    segmentation_resized_list = []
    for depth, image, segmentation, out_K_inv_dot_xy_1 in zip(depth_list, image_list, segmentation_list, out_K_inv_dot_xy_1_list):
        depth = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        image = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        segmentation = cv2.resize(segmentation, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

        depth_resiezed_list.append(depth)
        image_resized_list.append(image)
        segmentation_resized_list.append(segmentation)

        # create face from segmentation
        faces = []
        for y in range(out_h-1):
            for x in range(out_w-1):
                segmentIndex = segmentation[y, x]
                # ignore non planar region
                if segmentIndex == nonplane_idx:
                    continue
                # add face if three pixel has same segmentatioin
                depths = [depth[y][x], depth[y + 1][x], depth[y + 1][x + 1]]
                if np.abs(depths[0]-depths[1]) > 0.05 or np.abs(depths[0]-depths[2]) > 0.05 or np.abs(depths[1]-depths[2]) > 0.05:
                    pass
                else:
                    if segmentation[y + 1, x] == segmentIndex and segmentation[y + 1, x + 1] == segmentIndex and min(depths) > 0 and max(depths) < 10:
                        faces.append((x, y, x, y + 1, x + 1, y + 1))

                depths = [depth[y][x], depth[y][x + 1], depth[y + 1][x + 1]]
                if np.abs(depths[0]-depths[1]) > 0.05 or np.abs(depths[0]-depths[2]) > 0.05 or np.abs(depths[1]-depths[2]) > 0.05:
                    pass
                else:
                    if segmentation[y][x + 1] == segmentIndex and segmentation[y + 1][x + 1] == segmentIndex and min(depths) > 0 and max(depths) < 10:
                        faces.append((x, y, x + 1, y + 1, x + 1, y))
        faces_list.append(faces)
        faces_num_list.append(len(faces))

    with open(folder + '/'+ f'planar_{suffix}.ply', 'w') as f:
        header = """ply
format ascii 1.0
comment VCGLIB generated
element vertex """
        header += str(out_h * out_w * len(depth_list))
        header += """
property float x
property float y
property float z
property uint8 red 
property uint8 green
property uint8 blue 
element face """
        header += str(sum(faces_num_list))
        header += """
property list uchar int vertex_indices
end_header
"""
        f.write(header)
        for v in range(len(faces_list)):
            segmentation = segmentation_resized_list[v]
            depth = depth_resiezed_list[v]
            image = image_resized_list[v]
            out_K_inv_dot_xy_1 = out_K_inv_dot_xy_1_list[v]

            for y in range(out_h):
                for x in range(out_w):
                    segmentIndex = segmentation[y][x]
                    if segmentIndex == nonplane_idx:
                        f.write("0.0 0.0 0.0 0 0 0\n")
                        continue
                    ray = out_K_inv_dot_xy_1[:, y, x]
                    X, Y, Z = ray * depth[y, x]
                    if c2w_seq_list is not None:
                        c2w = c2w_seq_list[v]
                        X, Y, Z = np.dot(c2w[:3,:3], [X, Y, Z]) + c2w[:3, 3]
                    blue, green, red = image[y, x, 0], image[y, x, 1], image[y, x, 2]
                    f.write(str(X) + ' ' + str(Y) + ' ' + str(Z) + ' ' + str(red) + ' ' + str(green) + ' ' + str(blue) + '\n')
        
        for v in range(len(faces_list)):
            faces = faces_list[v]
            for face in faces:
                f.write('3 ')
                for c in range(3):
                    f.write(str(face[c * 2 + 1] * out_w + face[c * 2] + v * out_h * out_w) + ' ')
                # f.write('6 ')
                # for c in range(3):
                #     f.write(str(float(face[c * 2]) / out_w) + ' ' + str(1 - float(face[c * 2 + 1]) / out_h) + ' ')
                f.write('\n')
        f.close()
    return

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
