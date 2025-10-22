import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from plana3r.inference_plana3r import inference_plana3r
from plana3r.model_plana3r_v2 import Plana3rModel as v2
from plana3r.model_plana3r_v1 import Plana3rModelNaive as v1
from plana3r.utils.image import load_images, load_images_cameras
from plana3r.utils.misc import get_raster_cameras_simple
from plana3r.utils.inference import *
from planar_splatting.utils.model_util import rot_to_quat, quaternion_mult

import argparse

NUM_GPUS = torch.cuda.device_count()
torch.backends.cudnn.benchmark = True

def make_pairs_simple(img_path_list):
    img_num = len(img_path_list)
    pairs = []
    for i in range(0, img_num-1):
        pairs.append([img_path_list[i], img_path_list[i+1]])
    return pairs

def get_args_parser():
    parser = argparse.ArgumentParser('Inference', add_help=False)
    # model
    parser.add_argument('--model', default="v2", type=str, help="string containing the model to build")
    parser.add_argument('--model_path', default="checkpoints/plana3r_v2_official.pth", help='path of a starting checkpoint')
    parser.add_argument('--pose_type', default='simple', type=str)

    parser.add_argument('--img_dir', type=str, default='examples/view2_2', help='path to image folder')
    parser.add_argument('--intrinsic_path', type=str, default='', help='path to image folder')

    parser.add_argument('--plot_dir', type=str, default='exp_results/plana3r_demo')

    parser.add_argument('--merge_version', type=str, default='dev', help="version of merge function: dev or stable")
    parser.add_argument('--output_mode', type=str, default='pts3d', help="output mode: pts3d or mesh", choices=['pts3d', 'mesh'])

    return parser

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_dir = args.plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    # get img file names
    if len(args.img_dir) > 0:
        img_dir = args.img_dir
        if not os.path.exists(img_dir):
            raise ValueError(f'img dir {img_dir} does not exist')
        scene_name = os.path.basename(img_dir)
        plot_dir = os.path.join(plot_dir, f'{scene_name}_{args.model}')
        os.makedirs(plot_dir, exist_ok=True)
    else:
        raise ValueError('img dir is not specified')
    
    file_names = os.listdir(img_dir)
    img_names = []
    intrinsic_list_in = []
    for name in file_names:
        if '.png' in name or '.jpg' in name or '.JPG' in name:
            img_names.append(name)
        if 'intrinsic.txt' in name:
            intrinsic = np.loadtxt(os.path.join(img_dir, name)).astype(np.float32)[:3, :3] 
            intrinsic_list_in.append(intrinsic)
    if len(img_names) == 0:
        raise ValueError(f'no image found in {img_dir}')
    img_names.sort()
    img_path_list = [os.path.join(img_dir, name) for name in img_names]

    intrinsic_path = args.intrinsic_path
    if len(intrinsic_path) > 0:
        if not os.path.exists(intrinsic_path):
            raise ValueError(f'intrinsic path {intrinsic_path} does not exist')
        intrinsic = np.loadtxt(intrinsic_path).astype(np.float32)[:3, :3] 
        intrinsic_list = [intrinsic] * len(img_path_list)
    else:
        if args.model == 'v1': 
            if len(intrinsic_list_in) != len(img_path_list):
                raise ValueError('intrinsic path is not specified')
            else:
                intrinsic_list = intrinsic_list_in
        else:
            intrinsic  = np.eye(3).astype(np.float32)
            intrinsic_list = [intrinsic] * len(img_path_list)
    c2w_gt_list = [np.eye(4)] * len(img_path_list)

    images_list = load_images_cameras(
        img_path_list, 
        size=512,
        camera_intrinsics_list=intrinsic_list,
        camera_extrinsics_list=c2w_gt_list, 
    )
    all_pairs = [tuple(_) for _ in make_pairs_simple(images_list)]

    # load our model    
    model = eval(args.model)(
                pretrained_model_name_or_path='',
                pos_embed='RoPE100', 
                img_size=(512, 512), 
                head_type='linear', 
                output_mode='pts3d', 
                depth_mode=('linear', -100, 100), 
                enc_embed_dim=1024, 
                enc_depth=24, 
                enc_num_heads=16, 
                dec_embed_dim=768, 
                dec_depth=12, 
                dec_num_heads=12,
                pose_type=args.pose_type,
            ).to(device)
    loaded_weight = torch.load(args.model_path, map_location=device)['model']
    model.load_state_dict(loaded_weight, strict=True)
    model.to(device).eval()

    c2ref_pose = torch.eye(4).cuda()
    plane_center_world_list = []
    plane_radii_world_list = []
    plane_normal_world_list = []
    plane_rot_q_world_list = []

    img_raw_list = [cv2.imread(img_path) for img_path in img_path_list]
    intrinsic_raw_list = []
    c2w_seq_list = []

    height_raw, width_raw = cv2.imread(img_path_list[0]).shape[:2]
    resize_scale_h = height_raw / images_list[0]['img'].shape[-2] 
    resize_scale_w = width_raw / images_list[0]['img'].shape[-1] 

    use_pred_intrinsic = (args.model == 'v2')
    
    output_view1, output_view2 = inference_plana3r(
        all_pairs, 
        model, 
        device, 
        batch_size=1, 
        include_gt_geo=False, 
        use_pred_intrinsic=use_pred_intrinsic)

    c2ref_pose = torch.eye(4).cuda()
    plane_center_world_list = []
    plane_radii_world_list = []
    plane_normal_world_list = []
    plane_rot_q_world_list = []


    for idx, img_pair in enumerate(all_pairs):
        intrinsic_resized_view1 = output_view1['viewinfo_list'][idx].intrinsic
        intrinsic_resized_view2 = output_view2['viewinfo_list'][idx].intrinsic
        intrinsic_raw_view1 = intrinsic_resized_view1.clone()
        intrinsic_raw_view2 = intrinsic_resized_view2.clone()
        intrinsic_raw_view1[0] *= resize_scale_w
        intrinsic_raw_view1[1] *= resize_scale_h
        intrinsic_raw_view2[0] *= resize_scale_w
        intrinsic_raw_view2[1] *= resize_scale_h
        
        if len(intrinsic_raw_list) == 0:
            intrinsic_raw_list = [intrinsic_raw_view1, intrinsic_raw_view2]
        else:
            intrinsic_raw_list.append(intrinsic_raw_view2)
        
        # get pred plane parameters of each view     
        radii1 = output_view1['plane_radii_list'][idx]
        radii2 = output_view2['plane_radii_list'][idx]
        valid_plane_mask1 = radii1[:, 0] * radii1[:, 1] * 4 > 0.05**2
        valid_plane_mask2 = radii2[:, 0] * radii2[:, 1] * 4 > 0.05**2

        plane_radii = torch.cat(
            [
                output_view1['plane_radii_list'][idx][valid_plane_mask1],
                output_view2['plane_radii_list'][idx][valid_plane_mask2]
            ], dim=0).to(device)
        
        plane_center = torch.cat(
            [
                output_view1['plane_center_local_list'][idx][valid_plane_mask1],
                output_view2['plane_center_local_list'][idx][valid_plane_mask2]
            ], dim=0).to(device)     
        
        plane_normal = torch.cat(
            [
                output_view1['plane_normal_list'][idx][valid_plane_mask1],
                output_view2['plane_normal_list'][idx][valid_plane_mask2]
            ], dim=0).to(device) 

        plane_rot_q = torch.cat(
            [
                output_view1['plane_rot_q_normed_list'][idx][valid_plane_mask1],
                output_view2['plane_rot_q_normed_list'][idx][valid_plane_mask2]
            ], dim=0).to(device) 


        # convert plane center to world coordinate
        plane_center_homo = torch.cat([plane_center, torch.ones_like(plane_center[:, 0:1])], dim=-1)
        plane_center_world = (c2ref_pose @ plane_center_homo.t()).t()[..., :3]

        # convert plane rot_q to world coordinate
        ## convert c2fef_pose from matrix to quaternion
        c2ref_pose_quat = rot_to_quat(c2ref_pose[None]) # 1, 4
        ## convert plane_rot_q from local to world coordinate
        plane_rot_q_world = quaternion_mult(c2ref_pose_quat, plane_rot_q)
        
        # convert plane normal to world coordinate
        plane_normal_world = (c2ref_pose[:3, :3] @ plane_normal.t()).t()

        # update camera to renference pose
        rel_pose = output_view2['pred_c2w_list'][idx].cuda()
        c2ref_pose = c2ref_pose @ rel_pose
        if len(c2w_seq_list) == 0:
            c2w_seq_list = [output_view1['pred_c2w_list'][idx].cuda(), output_view2['pred_c2w_list'][idx].cuda()]
        else:
            c2w_seq_list.append(c2ref_pose)
        
        # store plane parameters
        plane_center_world_list.append(plane_center_world)
        plane_radii_world_list.append(plane_radii)
        plane_normal_world_list.append(plane_normal_world)
        plane_rot_q_world_list.append(plane_rot_q_world)

        # import pdb; pdb.set_trace()

    # merge plane primitives
    plane_insIDs, valid_mask = merge_primitives(
        plane_normal_world_list, 
        plane_center_world_list, 
        plane_radii_world_list, 
        plane_rot_q_world_list,
        merge_version=args.merge_version)
        
    print(f'save to {plot_dir}')
    draw_prims(torch.cat(plane_normal_world_list, dim=0)[valid_mask], 
               torch.cat(plane_center_world_list, dim=0)[valid_mask], 
               torch.cat(plane_radii_world_list, dim=0)[valid_mask],  
               torch.cat(plane_rot_q_world_list, dim=0)[valid_mask],
            #    plane_id=plane_insIDs[valid_mask],
               plot_dir=plot_dir,
               suffix='rawPrims',
               flip_3d=True)

    # get normlized color of primitives
    max_id = plane_insIDs.max()
    plane_insIDs_normlized = plane_insIDs.float() / max_id
    plane_insIDs_normlized_c3 = plane_insIDs_normlized[..., None].repeat(1, 3)

    # ---------------------------------------------------------------------------------------------------------------------------------
    # --- Note: we should rast seg instances to the resolution of the GT segmentation
    # ---------------------------------------------------------------------------------------------------------------------------------
    # get color_map
    colorMap_vis = get_random_color_map(12000)(10000)
    # get rast view info
    num_views = len(c2w_seq_list)
    view_info_ori_list = []
    for v in range(num_views):
        view_info_ori_view = get_raster_cameras_simple(intrinsic_raw_list[v], c2w_seq_list[v].cuda(), height=height_raw, width=width_raw)
        view_info_ori_list.append(view_info_ori_view)
    # rast on each view
    rgb_insIDs_list, allmap_list_raw = rast_primitives(
        view_info_ori_list, plane_center_world_list, plane_radii_world_list, plane_rot_q_world_list, 
        plane_insIDs_normlized_c3, plane_insIDs,
        height=height_raw, width=width_raw)
    # update plane primitives
    plane_center_updated, _, plane_rot_q_updated, _ = upadte_plane_parameters(
        plane_center_world_list, plane_normal_world_list, plane_rot_q_world_list, plane_radii_world_list, plane_insIDs)
    _, allmap_list = rast_primitives(
        view_info_ori_list, plane_center_updated, plane_radii_world_list, plane_rot_q_updated, 
        plane_insIDs_normlized_c3, plane_insIDs,
        height=height_raw, width=width_raw)

    # get seg_map and seg_param
    seg_map_global_list, seg_map_local_list, seg_params_list = [], [], []
    seg_masks_list = []
    plane_depth_list, pred_seg_list = [], []
    out_K_inv_dot_xy_1_list = []
    plot_h, plot_w = 192, 256
    # plot_h, plot_w = height_raw, width_raw
    for v in range(num_views): 
        seg_map_global, seg_map_local, seg_params, seg_masks = get_per_view_rast_segmap_segparam(
            allmap_list[v], view_info_ori_list[v], rgb_insIDs_list[v], plane_insIDs, min_mask_size=500)
        seg_map_global_list.append(seg_map_global)
        seg_map_local_list.append(seg_map_local)
        seg_params_list.append(seg_params)
        seg_masks_list.append(seg_masks)

        predMasks = torch.stack(seg_masks_list[v], dim=0).float()

        pred_param = torch.stack(seg_params_list[v], dim=0)  # normal * offset
        pred_offset = pred_param.norm(dim=-1, p=2, keepdim=True)
        pred_normal = F.normalize(pred_param, dim=-1)
        pred_param = pred_param / (pred_offset * pred_offset)

        # depth = offset / n \dot K^{-1}q
        xx = torch.arange(0, width_raw).reshape(1, -1).repeat(height_raw, 1)
        yy = torch.arange(0, height_raw).reshape(-1, 1).repeat(1, width_raw)
        ones = torch.ones_like(xx)
        uv1 = torch.stack([xx, yy, ones], dim=-1)  # 192, 256, 3
        uv1 = uv1.reshape(-1, 3).float().cuda()
        K_inv = torch.inverse(intrinsic_raw_list[v]).float().cuda()
        plane_depth = torch.zeros(height_raw, width_raw).cuda()
        plane_depth_rast = allmap_list[v][0:1].squeeze()
        for pidx, (offset, normal, seg_mask) in enumerate(zip(pred_offset, pred_normal, predMasks)):
            if offset == 0 or normal.sum() == 0:
                predMasks[pidx] = 0
                continue
            ray = K_inv @ uv1.T
            depth_i = offset / torch.mm(normal.reshape(1, 3), ray)
            depth_i = depth_i.reshape(height_raw, width_raw)
            mask = (seg_mask > 0)
            depth_i = depth_i * mask

            depth_i_rast = plane_depth_rast * mask
            invalid_depth_i_mask = (depth_i_rast - depth_i).abs() > 0.1
            valid_pix_num = (mask & (~invalid_depth_i_mask)).sum()
            valid_ratio = valid_pix_num / mask.sum()
            if valid_ratio < 0.6 or valid_pix_num < 20:
                predMasks[pidx] = 0
                continue
            depth_i = depth_i * (~invalid_depth_i_mask).float()
            predMasks[pidx] = predMasks[pidx] * (~invalid_depth_i_mask).float()
            plane_depth = plane_depth + depth_i
        assert len(predMasks) == len(pred_param)
        pred_seg = torch.zeros_like(plane_depth)
        for pidx in range(len(predMasks)):
            pred_seg[predMasks[pidx]>0] = pidx + 1
        pred_seg = pred_seg.int().cpu().numpy().astype(np.int32)
        plane_depth = plane_depth.cpu().numpy()
        
        invalid_depth_mask = np.abs(plane_depth - plane_depth_rast.cpu().numpy()) > 0.1
        plane_depth[invalid_depth_mask] = 0

        # save seg map
        plot_segmentation(pred_seg, img_raw_list[v], plot_dir, suffix=f'{v}')

        # save input image
        img_path = img_path_list[v]
        cv2.imwrite(os.path.join(plot_dir, f'view_{v}.jpg'), cv2.imread(img_path))

        # save intrinsic
        intrinsic_raw = intrinsic_raw_list[v].cpu().numpy()
        np.savetxt(os.path.join(plot_dir, f'view_{v}_intrinsic.txt'), intrinsic_raw)
        basename = img_path.split('/')[-1].split('.')[0]
        np.savetxt(os.path.join(img_dir, f'{basename}_intrinsic.txt'), intrinsic_raw)

        out_K_inv_dot_xy_1 = get_coordinate_map(intrinsic_raw_list[v].cpu().numpy(), height_raw, width_raw, plot_h, plot_w, device)
        out_K_inv_dot_xy_1 = out_K_inv_dot_xy_1.cpu().numpy().reshape(3, plot_h, plot_w)
        # writePLYFile(plot_dir, f'{v}_planar', plane_depth, pred_seg, img_raw_list[v], 0, out_K_inv_dot_xy_1, plot_h, plot_w)
        
        plane_depth_list.append(plane_depth)
        pred_seg_list.append(pred_seg)
        out_K_inv_dot_xy_1_list.append(out_K_inv_dot_xy_1)
    
    c2w_seq_list = [c2w.cpu().numpy() for c2w in c2w_seq_list]
    if args.output_mode == 'mesh':
        writePLYFile(plot_dir, f'planar_rgb_{num_views}views', plane_depth_list, pred_seg_list, img_raw_list, 0, out_K_inv_dot_xy_1_list, plot_h, plot_w, c2w_seq_list)
        
        seg_map_color_list = []
        for seg_map_global in seg_map_global_list:
            seg_map_color = colorMap_vis[seg_map_global.int().reshape(-1).cpu().numpy()]
            seg_map_color = seg_map_color.reshape(seg_map_global.shape[0], seg_map_global.shape[1], 3).astype(np.uint8)  # h, w, 3
            seg_map_color_list.append(seg_map_color)
        writePLYFile(plot_dir, f'planar_seg_{num_views}views', plane_depth_list, pred_seg_list, seg_map_color_list, 0, out_K_inv_dot_xy_1_list, plot_h, plot_w, c2w_seq_list)
    else:
        writePCDFile(plot_dir, f'planar_rgb_{num_views}views', plane_depth_list, pred_seg_list, img_raw_list, 0, out_K_inv_dot_xy_1_list, plot_h, plot_w, c2w_seq_list, flip_3d=True)
        seg_map_color_list = []
        for seg_map_global in seg_map_global_list:
            seg_map_color = colorMap_vis[seg_map_global.int().reshape(-1).cpu().numpy()]
            seg_map_color = seg_map_color.reshape(seg_map_global.shape[0], seg_map_global.shape[1], 3).astype(np.uint8)  # h, w, 3
            seg_map_color_list.append(seg_map_color)
        writePCDFile(plot_dir, f'planar_seg_{num_views}views', plane_depth_list, pred_seg_list, seg_map_color_list, 0, out_K_inv_dot_xy_1_list, plot_h, plot_w, c2w_seq_list, flip_3d=True)
