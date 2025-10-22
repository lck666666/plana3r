import os
import sys
sys.path.append('planar_splatting')
from planar_splatting.utils import loss_util,model_util
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from plana3r.utils.plot import save_normal_tensor_as_png, save_depth_tensor_as_png
from plana3r.utils.misc import make_batch_symmetric
from PIL import Image

def compute_gradient_magnitude(normal_gt):
    # import time
    # t1 = time.time()
    if normal_gt.dim() == 3:  # h, w, 3
        normal_gt_permuted = normal_gt.permute(2, 0, 1).contiguous()
        gradients = torch.gradient(normal_gt_permuted, dim=(1, 2))
        gradient_y, gradient_x = gradients  # 3, h, w
        magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        total_magnitude = magnitude.sum(dim=0)
    elif normal_gt.dim() == 4:  # b, h, w, 3
        normal_gt_permuted = normal_gt.permute(0,3,1,2).contiguous()
        gradients = torch.gradient(normal_gt_permuted, dim=(2, 3))
        gradient_y, gradient_x = gradients  # b, 3, h, w
        magnitude = torch.sqrt(gradient_x**2 + gradient_y**2)
        total_magnitude = magnitude.sum(dim=1)
    else:
        raise NotImplementedError
    
    # print('mask time', time.time() - t1)
    # import pdb; pdb.set_trace()
    return total_magnitude

def generate_gradient_mask(normal_gt, ratio=0.1):
    grad_magnitude = compute_gradient_magnitude(normal_gt)
    mask = torch.zeros_like(grad_magnitude, dtype=torch.bool)
    mask[grad_magnitude > ratio] = 1
    return mask
 
 # depth smooth loss（TV Loss）
def tv_loss(x, smooth_weight=0.5):
    # batch_size, height, width = x.shape
    dx = torch.abs(x[:, :, :-1] - x[:, :, 1:])  
    dy = torch.abs(x[:, :-1, :] - x[:, 1:, :])  
    return smooth_weight*(dx.mean() + dy.mean()) / 2  # Avg. Total Variation）

def loss_of_one_batch_plane(batch, model, device, symmetrize_batch=False, use_amp=False, ret=None, epoch=0, itr=-1, no_rast_epochs=1, criterion=None):
    weight_plane_normal = 1
    weight_plane_depth = 2
    weight_pose_loss = 10
    
    weight_hypersim = 0.01
    max_depth = 10.0
    view1, view2 = batch
    ignore_keys = set(['img_name','img_path','depthmap', 'dataset', 'label', 'instance', 'idx', 'view_idx', 'true_shape', 'rng', 'resolution','depthmap_resized', 'resolution'])
    for view in batch:
        for name in view.keys():  # pseudo_focal
            if name in ignore_keys:
                continue
            view[name] = view[name].to(device, non_blocking=True)
    if symmetrize_batch:
        view1, view2 = make_batch_symmetric(batch)

    output_folder = f'./outputs/{epoch}'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
    with torch.amp.autocast(device_type='cuda', enabled=bool(use_amp)):
        if epoch <= no_rast_epochs - 1:
            res_view1, res_view2, pred_rel_trans, pred_rel_rot_q = model(view1, view2, warmup=True)
        else:
            res_view1, res_view2, pred_rel_trans, pred_rel_rot_q = model(view1, view2, warmup=False)

        pred_planes = [res_view1["allmaps_list"], res_view2["allmaps_list"]]
        pred_planes_low = [res_view1["allmaps_list_low"], res_view2["allmaps_list_low"]]
        pred_planes_high = [res_view1["allmaps_list_high"], res_view2["allmaps_list_high"]]
        pred_depths_low = [res_view1["depth_patches_list_low"], res_view2["depth_patches_list_low"]]
        pred_normals_low = [res_view1["normal_patches_list_low"], res_view2["normal_patches_list_low"]]
        pred_depths_high = [res_view1["depth_patches_list_high"], res_view2["depth_patches_list_high"]]
        pred_normals_high = [res_view1["normal_patches_list_high"], res_view2["normal_patches_list_high"]]
        loss_final = 0.
        loss_depth = 0.
        loss_depth_patches = 0.
        loss_normal_patches = 0.
        loss_normal_l1 = 0.
        loss_normal_cos = 0.
        loss_plane = 0.
        bs = len(pred_depths_high[0])*2
        loss_pose_trans = 0.
        loss_pose_rot = 0.
        loss_rte = 0.
        loss_plane_depth_smooth_dict = 0.
        loss_patch_depth_high_smooth_dict = 0.

        if 'fov' in res_view1 and 'fov' in res_view2:
            gt_camera_intrinsics1 = view1['camera_intrinsics']
            gt_camera_intrinsics2 = view2['camera_intrinsics']
            gt_fx1 = gt_camera_intrinsics1[:, 0, 0]
            gt_fy1 = gt_camera_intrinsics1[:, 1, 1]
            gt_fx2 = gt_camera_intrinsics2[:, 0, 0]
            gt_fy2 = gt_camera_intrinsics2[:, 1, 1]
            H1, W1 = view1['true_shape'][:, 0].cuda(), view1['true_shape'][:, 1].cuda()
            H2, W2 = view2['true_shape'][:, 0].cuda(), view2['true_shape'][:, 1].cuda()
            gt_fov_h1 = 2 * torch.atan((H1 / 2) / gt_fy1)
            gt_fov_w1 = 2 * torch.atan((W1 / 2) / gt_fx1)
            gt_fov_h2 = 2 * torch.atan((H2 / 2) / gt_fy2)
            gt_fov_w2 = 2 * torch.atan((W2 / 2) / gt_fx2)
            gt_fov1 = torch.stack([gt_fov_h1, gt_fov_w1], dim=-1)
            gt_fov2 = torch.stack([gt_fov_h2, gt_fov_w2], dim=-1)
            loss_fov = (gt_fov1 - res_view1['fov']).abs().mean() + (gt_fov2 - res_view2['fov']).abs().mean()
        else:
            loss_fov = 0.0

        for view_id, (pred_trans, pred_rot_q) in enumerate(zip(pred_rel_trans, pred_rel_rot_q)):
            img0_c2w = view1['c2w'][view_id]
            img1_c2w = view2['c2w'][view_id]
            pair_rel_gt = img0_c2w.inverse()@ img1_c2w
            pair_rel_trans_gt = pair_rel_gt[:3,3]
            pair_rel_rot_gt = pair_rel_gt[:3,:3].unsqueeze(0)
            pair_rel_q_gt = model_util.rot_to_quat(pair_rel_rot_gt).squeeze()
            
            loss_trans = F.mse_loss(pred_trans, pair_rel_trans_gt)
            loss_rot =  F.mse_loss(pred_rot_q, pair_rel_q_gt)
            
            pred_trans = pred_trans.unsqueeze(0).float()  # (1, 3)
            pair_rel_trans_gt = pair_rel_trans_gt.unsqueeze(0).float()  # (1, 3)

            cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
            similarity = cos_sim(pred_trans, pair_rel_trans_gt)  # output: (1,)

            rte = 1 - similarity.mean()
            
            if not torch.isnan(rte) and not torch.isinf(rte):
                loss_rte += rte
                
            if not torch.isnan(loss_trans) and not torch.isinf(loss_trans) and not torch.isnan(loss_rot) and not torch.isinf(loss_rot):
                loss_pose_trans += loss_trans
                loss_pose_rot += loss_rot
            
            if torch.isnan(loss_trans) or torch.isinf(loss_trans):
                print("pred_trans: ",pred_trans)
                print("gt_trans: ",pair_rel_trans_gt)
                print("image_path: ", view1['img_path'][view_id])
                print("image_path: ", view2['img_path'][view_id])
            if torch.isnan(loss_rot)or torch.isinf(loss_rot):
                print("pred_rot: ",pred_rot_q)
                print("gt_rot: ",pair_rel_q_gt)
                print("img0_c2w: ",img0_c2w)
                print("img1_c2w: ",img1_c2w)
                print("pair_rel_gt: ",pair_rel_gt)
                print("image_path: ", view1['img_path'][view_id])
                print("image_path: ", view2['img_path'][view_id])
                
        for view_id, (pred_depth_low, pred_depth_high) in enumerate(zip(pred_depths_low, pred_depths_high)):
            for i, (depth_low, depth_high) in enumerate(zip(pred_depth_low,pred_depth_high)):
                if depth_high.shape[0] == 3072:
                    h, w = 48, 64
                    depth_high_2d = depth_high.view(h, w).unsqueeze(0)
                loss_patch_depth_high_smooth = tv_loss(depth_high_2d, smooth_weight=1)
                
                depth_low = depth_low.view(-1)
                depth_high = depth_high.view(-1)
                if view_id == 0:
                    view_info = res_view1["viewinfo_list"][i]
                else:
                    view_info = res_view2["viewinfo_list"][i]
                
                valid_depth_mask_low = view_info.patch_depth_low.abs() > 0
                valid_depth_mask_low = valid_depth_mask_low.view(-1)
                
                valid_depth_mask_high = view_info.patch_depth_high.abs() > 0
                valid_depth_mask_high = valid_depth_mask_high.view(-1)
                
                gt_depth_low = view_info.patch_depth_low.view(-1)
                gt_depth_high = view_info.patch_depth_high.view(-1)

                loss_patch_depth_low = loss_util.metric_depth_loss(depth_low, gt_depth_low, valid_depth_mask_low, max_depth=max_depth)
                loss_patch_depth_high = loss_util.metric_depth_loss(depth_high, gt_depth_high, valid_depth_mask_high, max_depth=max_depth)
                
                if 'hypersim' in view_info.image_path:
                    loss_depth_patches += (loss_patch_depth_low + loss_patch_depth_high) * weight_plane_depth * 10 * weight_hypersim
                else:
                    loss_depth_patches += (loss_patch_depth_low + loss_patch_depth_high) * weight_plane_depth * 10 + loss_patch_depth_high_smooth * weight_plane_depth 
                
                loss_patch_depth_high_smooth_dict += loss_patch_depth_high_smooth * weight_plane_depth 
                
                if torch.isnan(loss_depth_patches):
                    print("image_path: ", view_info.image_path)
                    print("pred depth_low: ", depth_low.min().item(), "~", depth_low.max().item())
                    print("pred depth_high: ", depth_high.min().item(), "~", depth_high.max().item())
                    print("pred gt_depth_low: ", gt_depth_low.min().item(), "~", gt_depth_low.max().item())
                    print("pred gt_depth_high: ", gt_depth_high.min().item(), "~", gt_depth_high.max().item())
        
        for view_id, (pred_normal_low, pred_normal_high) in enumerate(zip(pred_normals_low, pred_normals_high)):
            for i, (normal_low, normal_high) in enumerate(zip(pred_normal_low,pred_normal_high)):
                normal_low = normal_low.view(-1, 3)
                normal_high = normal_high.view(-1, 3)
                
                if view_id == 0:
                    view_info = res_view1["viewinfo_list"][i]
                else:
                    view_info = res_view2["viewinfo_list"][i]
                
                gt_normal_low = view_info.patch_normal_low
                valid_normal_mask_low = gt_normal_low.abs().sum(dim=-1) > 0
                valid_normal_mask_low = valid_normal_mask_low.view(-1)
                
                gt_normal_high = view_info.patch_normal_high
                valid_normal_mask_high = gt_normal_high.abs().sum(dim=-1) > 0
                valid_normal_mask_high = valid_normal_mask_high.view(-1)
                
                loss_plane_normal_l1_low, loss_plane_normal_cos_low = loss_util.normal_loss(normal_low, gt_normal_low.view(-1, 3), valid_normal_mask_low)
                loss_plane_normal_l1_high, loss_plane_normal_cos_high = loss_util.normal_loss(normal_high, gt_normal_high.view(-1, 3), valid_normal_mask_high)
            
                if 'hypersim' in view_info.image_path:
                    loss_normal_patches += (loss_plane_normal_l1_low +  loss_plane_normal_l1_high) * weight_plane_normal * 5 * weight_hypersim
                    loss_normal_patches += (loss_plane_normal_cos_low + loss_plane_normal_cos_high) * weight_plane_normal * 5 * weight_hypersim
                else:
                    loss_normal_patches += (loss_plane_normal_l1_low +  loss_plane_normal_l1_high) * weight_plane_normal * 5
                    loss_normal_patches += (loss_plane_normal_cos_low + loss_plane_normal_cos_high) * weight_plane_normal * 5
                
                 
                if torch.isnan(loss_normal_patches):
                    print("image_path: ", view_info.image_path)
                    print("pred normal_low: ", normal_low.min().item(), "~", normal_low.max().item())
                    print("pred normal_high: ", normal_high.min().item(), "~", normal_high.max().item())
                    print("pred gt_normal_low: ", gt_normal_low.min().item(), "~", gt_normal_low.max().item())
                    print("pred gt_normal_high: ", gt_normal_high.min().item(), "~", gt_normal_high.max().item())
                
        if epoch > no_rast_epochs - 1:
            for view_id, (pred_plane_high,pred_plane_low,pred_plane, pred_depth_low, pred_normal_low, pred_depth_high, pred_normal_high) in enumerate(zip(pred_planes_high,pred_planes_low,pred_planes, pred_depths_low, pred_normals_low, pred_depths_high, pred_normals_high)):
                for i, (allmap_high,allmap_low, allmap, depth_patches_low, normal_patches_low, depth_patches_high, normal_patches_high) in enumerate(zip(pred_plane_high,pred_plane_low,pred_plane, pred_depth_low, pred_normal_low, pred_depth_high, pred_normal_high)):
                    depth_rast = allmap[0:1].squeeze().view(-1)
                    normal_local_ = allmap[2:5]
                    
                    depth_rast_low = allmap_low[0:1].squeeze().view(-1)
                    normal_local_low_ = allmap_low[2:5]
                    
                    depth_rast_high = allmap_high[0:1].squeeze().view(-1)
                    normal_local_high_ = allmap_high[2:5]
                    
                    if view_id == 0:
                        view_info = res_view1["viewinfo_list"][i]
                    else:
                        view_info = res_view2["viewinfo_list"][i]
                      
                    #normal_global = (normal_local_.permute(1,2,0) @ (raster_cam_w2c[:3,:3].T)).view(-1, 3)
                    # ------------ get aux maps
                    vis_weight = allmap[1:2].squeeze().view(-1)
                    valid_ray_mask = vis_weight > 0.00001
                    valid_normal_mask = view_info.mono_normal_local.abs().sum(dim=-1) > 0
                    valid_normal_mask = valid_normal_mask.view(-1)
                    valid_depth_mask = view_info.mono_depth.abs() > 0
                    valid_depth_mask = valid_depth_mask.view(-1)
                    valid_ray_mask = valid_ray_mask & valid_depth_mask & valid_normal_mask
                    
                    vis_weight_low = allmap_low[1:2].squeeze().view(-1)
                    valid_ray_mask_low = vis_weight_low > 0.00001
                    valid_normal_mask = view_info.mono_normal_local.abs().sum(dim=-1) > 0
                    valid_normal_mask = valid_normal_mask.view(-1)
                    valid_depth_mask = view_info.mono_depth.abs() > 0
                    valid_depth_mask = valid_depth_mask.view(-1)
                    valid_ray_mask_low = valid_ray_mask_low & valid_depth_mask & valid_normal_mask
                    
                    vis_weight_high = allmap_high[1:2].squeeze().view(-1)
                    valid_ray_mask_high = vis_weight_high > 0.00001
                    valid_normal_mask = view_info.mono_normal_local.abs().sum(dim=-1) > 0
                    valid_normal_mask = valid_normal_mask.view(-1)
                    valid_depth_mask = view_info.mono_depth.abs() > 0
                    valid_depth_mask = valid_depth_mask.view(-1)
                    valid_ray_mask_high = valid_ray_mask_high & valid_depth_mask & valid_normal_mask
                    # ======================================= calculate losses
                    # ------------ calculate plane loss
                    # normal_gt = view_info.mono_normal_local
                    # boundry_mask = generate_gradient_mask(normal_gt, ratio=0.1)
                    #boundry_mask_np = (boundry_mask * 255).cpu().numpy().astype(np.uint8)
                    #img = Image.fromarray(boundry_mask_np, mode='L')
                    #img.save("outputs_ply/boundry.png")
                    
                    normal_local_rast = normal_local_.permute(1,2,0).view(-1, 3)
                    normal_local_rast_low = normal_local_low_.permute(1,2,0).view(-1, 3)
                    normal_local_rast_high = normal_local_high_.permute(1,2,0).view(-1, 3)
                    
                    
                    loss_plane_normal_l1, loss_plane_normal_cos = loss_util.normal_loss(normal_local_rast, view_info.mono_normal_local.view(-1, 3), valid_ray_mask)
                    loss_plane_normal_low_l1, loss_plane_normal_low_cos = loss_util.normal_loss(normal_local_rast_low, view_info.mono_normal_local.view(-1, 3), valid_ray_mask_low)
                    loss_plane_normal_high_l1, loss_plane_normal_high_cos = loss_util.normal_loss(normal_local_rast_high, view_info.mono_normal_local.view(-1, 3), valid_ray_mask_high)
                    
                    loss_normal_l1 += ((loss_plane_normal_l1 + loss_plane_normal_low_l1 + loss_plane_normal_high_l1) * weight_plane_normal)/3.0
                    loss_normal_cos += ((loss_plane_normal_cos + loss_plane_normal_low_cos + loss_plane_normal_high_cos)* weight_plane_normal)/3.0
                    
                    loss_plane_depth = loss_util.metric_depth_loss(depth_rast, view_info.mono_depth.view(-1), valid_ray_mask, max_depth=max_depth)
                    loss_plane_depth_smooth = tv_loss(allmap[0:1])
                    
                    loss_plane_depth_low = loss_util.metric_depth_loss(depth_rast_low, view_info.mono_depth.view(-1), valid_ray_mask_low, max_depth=max_depth)
                    loss_plane_depth_smooth_low = tv_loss(allmap_low[0:1])
                    
                    loss_plane_depth_high = loss_util.metric_depth_loss(depth_rast_high, view_info.mono_depth.view(-1), valid_ray_mask_high, max_depth=max_depth)
                    loss_plane_depth_smooth_high = tv_loss(allmap_high[0:1])
                    
                    if 'hypersim' in view_info.image_path:
                        loss_normal_l1 *= weight_hypersim
                        loss_normal_cos *= weight_hypersim
                        loss_plane_depth *= weight_hypersim
                        loss_plane_depth_smooth *= weight_hypersim

                    
                    #visualization and save some training results
                    if itr % 2000 == 0:
                    #if itr % 1 == 0:
                        if view_id == 0:
                            view_tag = 'view1'
                        else:
                            view_tag = 'view2'
                        img_path = view_info.image_path
                        img_name = img_path.split('/', 2)[-1].replace('.jpg', '').replace('/', '_')
                        normal_np = ((allmap[2:5].clone() + 1)/2 * 255).clamp(0,255).detach().cpu().numpy()
                        normal_np = np.transpose(normal_np, (1, 2, 0)).astype(np.uint8)
                        image_pil = Image.fromarray(normal_np, mode='RGB')

                        image_pil.save(f'{output_folder}/normal_ep{epoch}_itr{itr}_{img_name}_rast_{view_tag}_p{i}.png')
                        gt_depth = view_info.patch_depth_low.view(-1)
                        save_depth_tensor_as_png(depth_patches_low.squeeze().view(-1), f"{output_folder}/depth_patch_{epoch}_itr{itr}_{img_name}_predict_low_{view_tag}_p{i}.png")
                        save_depth_tensor_as_png(gt_depth, f"{output_folder}/depth_patch_{epoch}_itr{itr}_{img_name}_gt_low_{view_tag}_p{i}.png")
                        
                        gt_depth = view_info.patch_depth_high.view(-1)
                        save_depth_tensor_as_png(depth_patches_high.squeeze().view(-1), f"{output_folder}/depth_patch_{epoch}_itr{itr}_{img_name}_predict_high_{view_tag}_p{i}.png")
                        save_depth_tensor_as_png(gt_depth, f"{output_folder}/depth_patch_{epoch}_itr{itr}_{img_name}_gt_high_{view_tag}_p{i}.png")
                        
                        gt_depth_img_np =  view_info.mono_depth.cpu().numpy()
                        gt_depth_img_np[gt_depth_img_np > max_depth] = max_depth
                        min_val, max_val = gt_depth_img_np.min(), gt_depth_img_np.max()
                        if max_val - min_val == 0:
                            depth_normalized = np.zeros_like(gt_depth_img_np, dtype=np.uint8)  
                        else:
                            depth_normalized = ((gt_depth_img_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        Image.fromarray(depth_normalized, mode='L').save(f"{output_folder}/depth_{epoch}_itr{itr}_{img_name}_gt_{view_tag}_p{i}.png")
                        
                        gt_normal_low = view_info.patch_normal_low.view(-1, 3)
                        save_normal_tensor_as_png(normal_patches_low.squeeze().view(-1, 3), f"{output_folder}/normal_patch_{epoch}_itr{itr}_{img_name}_predict_low_{view_tag}_p{i}.png")
                        save_normal_tensor_as_png(gt_normal_low, f"{output_folder}/normal_patch_{epoch}_itr{itr}_{img_name}_gt_low_{view_tag}_p{i}.png")
                        gt_normal_img_np =  view_info.mono_normal_local.cpu().numpy()
                        normalized = (gt_normal_img_np + 1) * 127.5  # [-1,1] → [0,255]
                        normalized = np.clip(normalized, 0, 255)  
                        normalized = normalized.astype(np.uint8)
                        # 保存为RGB图像
                        Image.fromarray(normalized, 'RGB').save(f"{output_folder}/normal_ep{epoch}_itr{itr}_{img_name}_gt_{view_tag}_p{i}.png")  
                        
                        gt_normal_high = view_info.patch_normal_high.view(-1, 3)
                        save_normal_tensor_as_png(normal_patches_high.squeeze().view(-1, 3), f"{output_folder}/normal_patch_{epoch}_itr{itr}_{img_name}_predict_high_{view_tag}_p{i}.png")
                        save_normal_tensor_as_png(gt_normal_high, f"{output_folder}/normal_patch_{epoch}_itr{itr}_{img_name}_gt_high_{view_tag}_p{i}.png")
                        
                    if torch.isnan(loss_plane_depth_smooth):
                        loss_plane_depth_smooth = 0
                        print("image causes smooth loss nan: ",img_name)
                        
                    if torch.isnan(loss_plane_depth):
                        loss_plane_depth = 0
                        img_path = view_info.image_path
                        img_name = img_path.split('/', 2)[-1].replace('.jpg', '').replace('/', '_')
                        print("image causes nan: ",img_name)
                        print("pred depth range:", depth_rast.min().item(), "~", depth_rast.max().item())
                        print("gt depth range:", view_info.mono_depth.view(-1).min(), "~", view_info.mono_depth.view(-1).max())
                        #raise RuntimeError("NaN detected in loss_plane_depth! Check depth prediction or ground truth.")
                    #if True:
                    if torch.isnan(loss_plane_depth_low):
                        loss_plane_depth_low = 0
                        img_path = view_info.image_path
                        img_name = img_path.split('/', 2)[-1].replace('.jpg', '').replace('/', '_')
                        print("image causes nan: ",img_name)
                        print("pred depth range:", depth_rast.min().item(), "~", depth_rast.max().item())
                        print("gt depth range:", view_info.mono_depth.view(-1).min(), "~", view_info.mono_depth.view(-1).max())
                    
                    if torch.isnan(loss_plane_depth_high):
                        loss_plane_depth_high = 0
                        img_path = view_info.image_path
                        img_name = img_path.split('/', 2)[-1].replace('.jpg', '').replace('/', '_')
                        print("image causes nan: ",img_name)
                        print("pred depth range:", depth_rast.min().item(), "~", depth_rast.max().item())
                        print("gt depth range:", view_info.mono_depth.view(-1).min(), "~", view_info.mono_depth.view(-1).max())
                           
                    if torch.isnan(loss_normal_l1):
                        img_path = view_info.image_path
                        img_name = img_path.split('/', 2)[-1].replace('.jpg', '').replace('/', '_')
                        print("image causes nan: ",img_name)
                        loss_normal_l1 = 0
                        gt_depth_img_np =  view_info.mono_depth.cpu().numpy()
                        min_val, max_val = gt_depth_img_np.min(), gt_depth_img_np.max()
                        if max_val - min_val == 0:
                            depth_normalized = np.zeros_like(gt_depth_img_np, dtype=np.uint8)  
                        else:
                            depth_normalized = ((gt_depth_img_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                        # Image.fromarray(depth_normalized, mode='L').save(f"{output_folder}/gt_depth_{epoch}.png")
                        
                        gt_normal_img_np =  view_info.mono_normal_local.cpu().numpy()
                        normalized = (gt_normal_img_np + 1) * 127.5  # [-1,1] → [0,255]
                        normalized = np.clip(normalized, 0, 255)  # 防止溢出
                        normalized = normalized.astype(np.uint8)
                        # save rgb
                        # Image.fromarray(normalized, 'RGB').save(f"{output_folder}/gt_normal_{epoch}.png")    
                        print("pred normal range:", normal_local_rast.min().item(), "~", normal_local_rast.max().item())
                        print("gt normal range:", view_info.mono_normal_local.view(-1, 3).min(), "~", view_info.mono_normal_local.view(-1, 3).max())
                        
                        #raise RuntimeError("NaN detected in loss_plane_normal_l1! Check normal prediction or L1 calculation.")
                    if torch.isnan(loss_normal_cos):
                        img_path = view_info.image_path
                        img_name = img_path.split('/', 2)[-1].replace('.jpg', '').replace('/', '_')
                        print("image causes nan: ",img_name)
                        loss_normal_cos = 0
                        print("normal nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                        print("pred normal range:", normal_local_rast.min().item(), "~", normal_local_rast.max().item())
                        print("gt normal range:", view_info.mono_normal_local.view(-1, 3).min(), "~", view_info.mono_normal_local.view(-1, 3).max())
                        #raise RuntimeError("NaN detected in loss_plane_normal_l1! Check normal prediction or L1 calculation.")
                    
                    
                    loss_depth += ((loss_plane_depth + loss_plane_depth_low + loss_plane_depth_high)/3.0) * weight_plane_depth
                    loss_plane_depth_smooth_dict += (loss_plane_depth_smooth) * weight_plane_depth
                    loss_plane += ((loss_plane_depth + loss_plane_depth_low + loss_plane_depth_high)/3.0) * weight_plane_depth + ((loss_plane_depth_smooth+loss_plane_depth_smooth_low + loss_plane_depth_smooth_high)/3.0) * weight_plane_depth \
                                + loss_normal_l1 + loss_normal_cos
        
        if 'pts3d' in res_view1 and 'pts3d_in_other_view' in res_view2:
            loss_pts3d_with_conf = criterion(view1, view2, res_view1, res_view2) if criterion is not None else None
        else:
            loss_pts3d_with_conf = [0.]

        loss_final = loss_depth_patches / bs + loss_normal_patches / bs  + loss_pose_trans /(bs // 2) * weight_pose_loss + loss_rte /(bs // 2) + loss_pose_rot /(bs // 2) * weight_pose_loss + loss_plane / bs
        loss_final += loss_pts3d_with_conf[0] * 5.0 + loss_fov * 5.0

        loss_details = dict()
        loss_details['loss_depth_patches'] = loss_depth_patches / bs
        loss_details['loss_normal_patches'] = loss_normal_patches / bs
        loss_details['loss_depth_3_rasts_avg'] = loss_depth / bs
        loss_details['loss_depth_smooth'] = loss_plane_depth_smooth_dict / bs
        loss_details['loss_depth_patch_smooth'] = loss_patch_depth_high_smooth_dict / bs
        loss_details['loss_normal_l1_3_rasts_avg'] = loss_normal_l1 / bs
        loss_details['loss_normal_cos_3_rasts_avg'] = loss_normal_cos / bs
        loss_details['loss_pose_rte'] = loss_rte /(bs // 2)
        loss_details['loss_pose_trans'] = loss_pose_trans /(bs // 2)
        loss_details['loss_pose_rot'] = loss_pose_rot /(bs // 2)
        if 'pts3d' in res_view1 and 'pts3d_in_other_view' in res_view2:
            loss_details.update(loss_pts3d_with_conf[1])
        if 'fov' in res_view1 and 'fov' in res_view2:
            loss_details['loss_fov'] = loss_fov
        
    return loss_final, loss_details