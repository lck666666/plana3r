# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Plana3r model class
# --------------------------------------------------------
import torch
import os
import math
from torch import nn
from torch.nn import functional as F
import numpy as np

from .utils.misc import get_raster_cameras, normalize_quaternion
from .layers.croco3d import AsymmetricCroCo3DStereo

from diff_rect_rasterization import RectRasterizationSettings, RectRasterizer 

from plana3r.loss_plana3r import generate_gradient_mask
from plana3r.utils.misc import fov_to_focal, fov_to_intrinsic_matrix

from plana3r.heads.plane_head import PlaneHead
from plana3r.heads.upsample_head import UpsampleHeadSimple
from plana3r.heads.camera_head import CameraHeadSimple, CameraHeadSimple_FoV
from plana3r.heads.points3d_head import Pts3dMLP

import sys
sys.path.append('planar_splatting')
from planar_splatting.utils import plot_util, model_util

import logging
log_dir = 'outputs'
os.makedirs(log_dir, exist_ok=True)
log_path = os.path.join(log_dir, 'train.log')
if os.path.exists(log_path):
    open(log_path, 'w').close()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            filename=log_path,
            mode='w',  
            encoding='utf-8'
        )
    ]
)

inf = float('inf')

class Plana3rModel(nn.Module):
    def __init__(
        self,
        pretrained_model_name_or_path: str='',
        embed_dim: int = 1024,
        hidden_dim: int = 512,
        patch_size: int = 16,
        output_mode='pts3d',
        head_type='linear',
        depth_mode=('exp', -inf, inf),
        conf_mode=('exp', 1, inf),
        freeze='none',
        landscape_only=True,
        patch_embed_cls='PatchEmbedDust3R',
        pose_type='simple',
        **croco_kwargs
    ):
        super().__init__()
        self.patch_size = patch_size
        self.patch_embed_cls = patch_embed_cls
        self.hidden_dim = hidden_dim
        croco_kwargs['patch_size'] = patch_size
        self.bg = torch.tensor([0., 0., 0.]).cuda()
        self.plane_normal_standard = torch.tensor([0., 0., 1.]).reshape(1, 3).cuda()
        self.plot_dir = "outputs" 
        self.pose_type = pose_type
        if len(pretrained_model_name_or_path) > 0:
            self.encoder = AsymmetricCroCo3DStereo.from_pretrained(
                pretrained_model_name_or_path,
                output_mode=output_mode,
                head_type=head_type,
                depth_mode=depth_mode,
                conf_mode=conf_mode,
                freeze=freeze,
                landscape_only=landscape_only,
                patch_embed_cls=patch_embed_cls,
                **croco_kwargs
            ) 
            print('*'*40)
            print('loading from pretrained model weights')
        else:
            print('*'*40)
            print('training without pretrained mode weights')
            self.encoder = AsymmetricCroCo3DStereo(
                output_mode=output_mode,
                head_type=head_type,
                depth_mode=depth_mode,
                conf_mode=conf_mode,
                freeze=freeze,
                landscape_only=landscape_only,
                patch_embed_cls=patch_embed_cls,
                **croco_kwargs
            )

        self.dec_dim = 768
        self.upsample_head = UpsampleHeadSimple(in_channels=self.dec_dim, out_channels=self.dec_dim)
        self.plane_head = PlaneHead(dec_dim=self.dec_dim, hidden_dim=self.hidden_dim)
        self.camera_head = CameraHeadSimple(dec_dim=self.dec_dim, hidden_dim=self.hidden_dim)
        self.camera_head_fov = CameraHeadSimple_FoV(dec_dim=self.dec_dim, hidden_dim=self.hidden_dim)

        # self.points3d_head = LinearPts3d(net=self.encoder, has_conf=True) 
        self.points3d_head = Pts3dMLP(dec_dim=self.dec_dim, hidden_dim=self.hidden_dim, net=self.encoder, has_conf=True) 

        self.pixel_coords_dict = {}
        self.uv = None

    def get_primitives(self, pred, input_view, view_tag, c2w, warm_up=False, split_threshold=0.5):
        '''
        c2w: b, 4, 4
        '''
        pred_plane_depth = pred[f'pred_center_depth_{view_tag}']  # b, n, 1
        pred_plane_depth = torch.clamp(pred_plane_depth, min=0.2, max=10.0) if not warm_up else pred_plane_depth
        bs, plane_num, _ = pred_plane_depth.shape

        pred_plane_rot_q_normed = normalize_quaternion(pred[f'pred_rot_{view_tag}'])  # b, n, 4
        pred_plane_rot_matrix = model_util.quat_to_rot(pred_plane_rot_q_normed.view(-1, 4))  # b*n, 3, 3
        pred_plane_normal = torch.bmm(pred_plane_rot_matrix, self.plane_normal_standard.reshape(1, 3, 1).expand(bs*plane_num, 3, 1)).squeeze(-1) # b*n, 3
        pred_plane_normal = pred_plane_normal.reshape(bs, plane_num, 3)  # b, n, 3

        intrinsics_resized = input_view['camera_intrinsics'] # b, 3, 3
        rgb_image_resized = input_view['img'] # b, 3, h, w
        pred_plane_normal_patches = torch.bmm(pred_plane_normal, c2w[:, :3, :3])  # b, n, 3

        pred_plane_radii = torch.clamp(pred[f'pred_radii_{view_tag}'], min=1e-3)  # b, n, 2
        pred_plane_radii_padding = torch.cat([pred_plane_radii, pred_plane_radii], dim=-1)  # b, n, 4

        
        if pred_plane_normal_patches.shape[1] == (rgb_image_resized.shape[2]*rgb_image_resized.shape[3])//(self.patch_size**2):
            h_patch = rgb_image_resized.shape[2]//self.patch_size
            w_patch = rgb_image_resized.shape[3]//self.patch_size
            scale_factor = self.patch_size
        elif pred_plane_normal_patches.shape[1] == (rgb_image_resized.shape[2]*rgb_image_resized.shape[3])*4//(self.patch_size**2):
            h_patch = rgb_image_resized.shape[2]*2//self.patch_size
            w_patch = rgb_image_resized.shape[3]*2//self.patch_size
            scale_factor = self.patch_size//2
        else:
            raise NotImplementedError

        intrinsics_patches = intrinsics_resized.clone() # b, 3, 3
        intrinsics_patches[:, :2, :] /= scale_factor # b, 3, 3
        
        pred_plane_normal_patches_map = pred_plane_normal_patches.view(bs, h_patch, w_patch, 3)
        boundry_mask = generate_gradient_mask(pred_plane_normal_patches_map, ratio=split_threshold)  # b, hp, wp
        
        pixel_coords_key = 'h%d_w%d'%(int(h_patch), int(w_patch))
        if pixel_coords_key in self.pixel_coords_dict:
            pixel_coords = self.pixel_coords_dict[pixel_coords_key]  # [3, n]
        else:
            y, x = torch.meshgrid(torch.arange(h_patch), torch.arange(w_patch), indexing='ij')
            pixel_coords = torch.stack([x, y, torch.ones_like(x)], dim=0)  # [3, hp, W]
            pixel_coords = pixel_coords.reshape(3, -1).cuda()  # [3, n]
            self.pixel_coords_dict[pixel_coords_key] = pixel_coords
        pixel_coords = pixel_coords.unsqueeze(0).repeat(bs, 1, 1).float()  # [bs, 3, n]
        camera_coords = torch.bmm(torch.linalg.inv(intrinsics_patches), pixel_coords)  # b, 3, n
        
        # Transform to world coordinates    
        camera_coords_depth = camera_coords * pred_plane_depth.permute(0, 2, 1)  # [b, 3, n]
        camera_coords_homo = torch.cat([camera_coords_depth, torch.ones_like(camera_coords_depth[:, 0:1])], dim=1)  # [b, 4, n]
        world_coords = torch.bmm(c2w, camera_coords_homo)  # [b, 4, n]
        pred_plane_center_world = world_coords.permute(0, 2, 1)[..., :3]  # [b, n, 3]
                  
        res = dict()
        res['boundry_mask'] = boundry_mask
        res['intrinsics_resized'] = intrinsics_resized
        res['pred_plane_rot_q_normed'] = pred_plane_rot_q_normed
        res['pred_plane_radii_padding'] = pred_plane_radii_padding
        res['pred_plane_center_world'] = pred_plane_center_world
        res['pred_plane_normal_patches'] = pred_plane_normal_patches
        res['pred_depth_patches'] = pred_plane_depth  # b, n, 1
        return res
    
    def get_view_info(self, view, c2ws, include_gt_geo=True):
        view_info_list = []
        bs = view['img'].shape[0]
        assert c2ws.dim() == 3  # b, 4, 4
        for b in range(bs):
            rgb_image_resized = view['img'][b]
            image_path = view['img_path'][b]
            intrinsics_resized = view['camera_intrinsics'][b]
            c2w = c2ws[b]
            if include_gt_geo:
                depthmap_resized = view['depthmap_resized'][b]
                mono_normal_resized = view['normalmap_resized'][b]
                view_idx = view['view_idx'][b]
                depthmap_patches = view['depthmap_patches'][b]
                normal_patches = view['normalmap_patches'][b]
                depthmap_patches_low = view['depthmap_patches_low'][b]
                normal_patches_low = view['normalmap_patches_low'][b]
                view_info = get_raster_cameras(intrinsics_resized,
                                                c2w,
                                                rgb_image_resized,
                                                depthmap_resized,
                                                mono_normal_resized,
                                                image_path,view_idx,
                                                depthmap_patches,
                                                normal_patches,
                                                depthmap_patches_low,
                                                normal_patches_low)
            else:
                view_info = get_raster_cameras(intrinsics_resized,c2w,rgb_image_resized,image_path=image_path)

            view_info_list.append(view_info)
        
        return view_info_list

    def render_primitives(self, view, viewinfo_list, pred_prim_low, pred_prim_high, training=False, warmup=False):
        allmaps_list = []
        allmaps_list_low = []  # for training
        allmaps_list_high = [] # for training

        final_plane_normal_list = []
        final_plane_center_world_list = []
        final_plane_radii_list = []
        final_plane_rot_q_normed_list = []

        bs = len(viewinfo_list)

        if not training:
            assert not warmup
        
        for b in range(bs):
            plane_center_world_low = pred_prim_low['pred_plane_center_world'][b]
            plane_radii_padding_low = pred_prim_low['pred_plane_radii_padding'][b]
            plane_rot_q_normed_low = pred_prim_low['pred_plane_rot_q_normed'][b]

            plane_center_world_high = pred_prim_high['pred_plane_center_world'][b]
            plane_radii_padding_high = pred_prim_high['pred_plane_radii_padding'][b]
            plane_rot_q_normed_high = pred_prim_high['pred_plane_rot_q_normed'][b]

            # select prim from low res
            boundary_mask_low = pred_prim_low['boundry_mask'][b]
            mask_low_flat = boundary_mask_low.flatten()
            selected_low_plane_center = plane_center_world_low[~mask_low_flat]   # keep False areas
            selected_low_plane_q = plane_rot_q_normed_low[~mask_low_flat]  
            selected_low_plane_radii = plane_radii_padding_low[~mask_low_flat]  
            
            # select prim from high res
            boundary_mask_high = F.interpolate(
                boundary_mask_low.float().unsqueeze(0).unsqueeze(0), 
                size=(boundary_mask_low.shape[0]*2, boundary_mask_low.shape[1]*2),
                mode='nearest'
            ).squeeze().bool()
            mask_high_flat = boundary_mask_high.flatten()
            selected_high_plane_center = plane_center_world_high[mask_high_flat]   # keep True areas
            selected_high_plane_q = plane_rot_q_normed_high[mask_high_flat]  
            selected_high_plane_radii = plane_radii_padding_high[mask_high_flat]  

            if warmup:
                # only use prim from high res
                plane_center_world_merged = plane_center_world_high
                plane_rot_q_normed_merged = plane_rot_q_normed_high
                plane_radii_padding_merged = plane_radii_padding_high
            else:
                # used combined prim
                plane_center_world_merged = torch.cat([selected_low_plane_center, selected_high_plane_center], dim=0)
                plane_rot_q_normed_merged = torch.cat([selected_low_plane_q, selected_high_plane_q], dim=0)
                plane_radii_padding_merged = torch.cat([selected_low_plane_radii, selected_high_plane_radii], dim=0)

                # used combined prim
                # print('Warning: using combined prim of low res')
                # plane_center_world_merged = plane_center_world_low
                # plane_rot_q_normed_merged = plane_rot_q_normed_low
                # plane_radii_padding_merged = plane_radii_padding_low

            primitive_num = plane_center_world_merged.shape[0]

            rgb_image_resized = view['img'][b]
            
            if not training:
                plane_rot_matrix_merged = model_util.quat_to_rot(plane_rot_q_normed_merged)  # n, 3, 3
                plane_normal_merged = torch.bmm(plane_rot_matrix_merged, self.plane_normal_standard.reshape(-1, 3, 1).expand(primitive_num, 3, 1)).squeeze(-1)
                
                final_plane_normal_list.append(plane_normal_merged)
                final_plane_center_world_list.append(plane_center_world_merged)
                final_plane_rot_q_normed_list.append(plane_rot_q_normed_merged)
                final_plane_radii_list.append(plane_radii_padding_merged)
            
            # get rast parameters
            view_info = viewinfo_list[b]
            tanfovx = view_info.tanfovx
            tanfovy = view_info.tanfovy
            raster_cam_w2c = view_info.raster_cam_w2c
            raster_cam_fullproj = view_info.raster_cam_fullproj
            raster_cam_center = view_info.raster_cam_center
            raster_img_center = view_info.raster_img_center
            splat_weight = 300.0
            
            # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
            screenspace_points = torch.zeros_like(plane_center_world_merged, dtype=plane_center_world_merged.dtype, requires_grad=True, device="cuda")
            try:
                screenspace_points.retain_grad()
            except:
                pass
            raster_settings = RectRasterizationSettings(
                                image_height=rgb_image_resized.shape[1],
                                image_width=rgb_image_resized.shape[2],
                                tanfovx=tanfovx,
                                tanfovy=tanfovy,
                                bg=self.bg,
                                scale_modifier=1.0,
                                viewmatrix=raster_cam_w2c,      #images-pair coordinates
                                projmatrix=raster_cam_fullproj, #images-pair coordinates
                                sh_degree=0,
                                campos=raster_cam_center,
                                prefiltered=False,
                                debug=False,
                                lambdaw=splat_weight * 5.0,
                                image_center=raster_img_center,
                                scales2=plane_radii_padding_merged[:, :2].detach())
            if not warmup:
                # ======================================= plane model forward
                rasterizer = RectRasterizer(raster_settings=raster_settings)
                rgb, _, allmap = rasterizer(
                            means3D = plane_center_world_merged, #images-pair coordinates
                            means2D = screenspace_points,
                            shs = None,
                            colors_precomp = torch.rand_like(plane_center_world_merged),
                            opacities = torch.ones_like(plane_center_world_merged)[:, :1],
                            scales = plane_radii_padding_merged,
                            rotations = plane_rot_q_normed_merged,
                            cov3D_precomp = None
                )
                allmaps_list.append(allmap)

                if training:
                    logging.info(f'primitive num: {primitive_num}')
                    rgb_np = rgb.clone().mul(255).byte().detach().cpu().numpy()
                    rgb_np = np.transpose(rgb_np, (1, 2, 0))
                    
                    # rast prim from low res
                    raster_settings_low = RectRasterizationSettings(
                                image_height=rgb_image_resized.shape[1],
                                image_width=rgb_image_resized.shape[2],
                                tanfovx=tanfovx,
                                tanfovy=tanfovy,
                                bg=self.bg,
                                scale_modifier=1.0,
                                viewmatrix=raster_cam_w2c,      #images-pair coordinates
                                projmatrix=raster_cam_fullproj, #images-pair coordinates
                                sh_degree=0,
                                campos=raster_cam_center,
                                prefiltered=False,
                                debug=False,
                                lambdaw=splat_weight * 5.0,
                                image_center=raster_img_center,
                                scales2=plane_radii_padding_low[:, :2].detach())
                    screenspace_points_low = torch.zeros_like(plane_center_world_low, dtype=plane_center_world_low.dtype, requires_grad=True, device="cuda")
                    try:
                        screenspace_points_low.retain_grad()
                    except:
                        pass
                    rasterizer_low = RectRasterizer(raster_settings=raster_settings_low)
                    _, _, allmap_low = rasterizer_low(
                                means3D = plane_center_world_low, #images-pair coordinates
                                means2D = screenspace_points_low,
                                shs = None,
                                colors_precomp = torch.rand_like(plane_center_world_low),
                                opacities = torch.ones_like(plane_center_world_low)[:, :1],
                                scales = plane_radii_padding_low,
                                rotations = plane_rot_q_normed_low,
                                cov3D_precomp = None
                    )
                    allmaps_list_low.append(allmap_low) #local coordinates
                    
                    # rast prim from high res
                    raster_settings_high = RectRasterizationSettings(
                                image_height=rgb_image_resized.shape[1],
                                image_width=rgb_image_resized.shape[2],
                                tanfovx=tanfovx,
                                tanfovy=tanfovy,
                                bg=self.bg,
                                scale_modifier=1.0,
                                viewmatrix=raster_cam_w2c,      #images-pair coordinates
                                projmatrix=raster_cam_fullproj, #images-pair coordinates
                                sh_degree=0,
                                campos=raster_cam_center,
                                prefiltered=False,
                                debug=False,
                                lambdaw=splat_weight * 5.0,
                                image_center=raster_img_center,
                                scales2=plane_radii_padding_high[:, :2].detach()
                    )
                    rasterizer_high = RectRasterizer(raster_settings=raster_settings_high)
                    screenspace_points_high = torch.zeros_like(plane_center_world_high, dtype=plane_center_world_high.dtype, requires_grad=True, device="cuda")
                    try:
                        screenspace_points_high.retain_grad()
                    except:
                        pass
                    _, _, allmap_high = rasterizer_high(
                                means3D = plane_center_world_high, #images-pair coordinates
                                means2D = screenspace_points_high,
                                shs = None,
                                colors_precomp = torch.rand_like(plane_center_world_high),
                                opacities = torch.ones_like(plane_center_world_high)[:, :1],
                                scales = plane_radii_padding_high,
                                rotations = plane_rot_q_normed_high,
                                cov3D_precomp = None
                    )
                    allmaps_list_high.append(allmap_high) #local coordinates

        res_dict = dict()
        res_dict["allmaps_list"] = allmaps_list
        if training:
            res_dict["allmaps_list_low"] = allmaps_list_low
            res_dict["allmaps_list_high"] = allmaps_list_high
            res_dict["depth_patches_list_low"] = [dp for dp in pred_prim_low['pred_depth_patches']]
            res_dict["depth_patches_list_high"] = [dp for dp in pred_prim_high['pred_depth_patches']]
            res_dict["normal_patches_list_low"] = [nop for nop in pred_prim_low['pred_plane_normal_patches']]
            res_dict["normal_patches_list_high"] = [nop for nop in pred_prim_high['pred_plane_normal_patches']]
        else:
            res_dict["plane_normal_list"] = final_plane_normal_list
            res_dict["plane_center_local_list"] = final_plane_center_world_list
            res_dict["plane_radii_list"] = final_plane_radii_list
            res_dict["plane_rot_q_normed_list"] = final_plane_rot_q_normed_list
        
        return res_dict

    def get_pts3d_from_prim(self, view1, view2, res_view1_dict, res_view2_dict, view_info1, view_info2):
        assert len(res_view1_dict['plane_normal_list']) == 1
        import pdb; pdb.set_trace()
        combined_plane_normal = torch.cat([res_view1_dict['plane_normal_list'][0], res_view2_dict['plane_normal_list'][0]], dim=0)
        combined_plane_center = torch.cat([res_view1_dict['plane_center_local_list'][0], res_view2_dict['plane_center_local_list'][0]], dim=0)
        combined_plane_radii = torch.cat([res_view1_dict['plane_radii_list'][0], res_view2_dict['plane_radii_list'][0]], dim=0)
        combined_plane_rot_q = torch.cat([res_view1_dict['plane_rot_q_normed_list'][0], res_view2_dict['plane_rot_q_normed_list'][0]], dim=0)

        image_height, image_width = view1['img'].shape[-2:]
        # get rast parameters
        viewinfo_list = [view_info1[0], view_info2[0]]
        allmaps_list = []
        for i in range (2):
            view_info = viewinfo_list[i]
            tanfovx = view_info.tanfovx
            tanfovy = view_info.tanfovy
            raster_cam_w2c = view_info.raster_cam_w2c
            raster_cam_fullproj = view_info.raster_cam_fullproj
            raster_cam_center = view_info.raster_cam_center
            raster_img_center = view_info.raster_img_center
            splat_weight = 300.0
            
            # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
            screenspace_points = torch.zeros_like(combined_plane_center, dtype=combined_plane_center.dtype, requires_grad=True, device="cuda")
            try:
                screenspace_points.retain_grad()
            except:
                pass
            raster_settings = RectRasterizationSettings(
                                image_height=image_height,
                                image_width=image_width,
                                tanfovx=tanfovx,
                                tanfovy=tanfovy,
                                bg=self.bg,
                                scale_modifier=1.0,
                                viewmatrix=raster_cam_w2c,      #images-pair coordinates
                                projmatrix=raster_cam_fullproj, #images-pair coordinates
                                sh_degree=0,
                                campos=raster_cam_center,
                                prefiltered=False,
                                debug=False,
                                lambdaw=splat_weight * 5.0,
                                image_center=raster_img_center,
                                scales2=combined_plane_radii[:, :2].detach())
            # ======================================= plane model forward
            rasterizer = RectRasterizer(raster_settings=raster_settings)
            rgb, _, allmap = rasterizer(
                        means3D = combined_plane_center, #images-pair coordinates
                        means2D = screenspace_points,
                        shs = None,
                        colors_precomp = torch.rand_like(combined_plane_center),
                        opacities = torch.ones_like(combined_plane_center)[:, :1],
                        scales = combined_plane_radii,
                        rotations = combined_plane_rot_q,
                        cov3D_precomp = None
            )
            allmaps_list.append(allmap)
        import pdb; pdb.set_trace()
        rast_depth1 = res_view1_dict['allmaps_list'][0][0] # h, w
        rast_depth2 = res_view2_dict['allmaps_list'][0][0] # h, w
        image_height, image_width = rast_depth1.shape

        if (self.uv is None) or (self.uv.shape[0] != image_height*image_width):
            uv = np.mgrid[0:image_height, 0:image_width].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float().cuda()
            uv = uv.reshape(2, -1).transpose(1, 0)  # h*w, 2            
            self.uv = uv
        else:
            uv =self.uv
        uv1 = torch.cat([uv, torch.ones_like(uv[:, 0:1])], dim=-1)  # hw, 3

        intrinsic1 = view1['camera_intrinsics'][0]
        intrinsic1_inv = torch.inverse(intrinsic1)  # 3, 3
        pts3d_local1 = ((intrinsic1_inv @ uv1.t()) * rast_depth1.reshape(1, -1)).transpose(1, 0)  # hw, 3   
        pts3d_world1 = pts3d_local1

        c2w = res_view2_dict['pred_c2w_list'][0] # 4,4
        intrinsic2 = view2['camera_intrinsics'][0]
        intrinsic2_inv = torch.inverse(intrinsic2)  # 3, 3
        pts3d_local2 = ((intrinsic2_inv @ uv1.t()) * rast_depth2.reshape(1, -1)).transpose(1, 0)  # hw, 3   
        pts3d_local_homo2 = torch.cat([pts3d_local2, torch.ones_like(pts3d_local2[:,0:1])], dim=-1)
        pts3d_world2 = torch.matmul(c2w.cuda(), pts3d_local_homo2.t())
        pts3d_world2 = pts3d_world2.t()[:, :3] # npts, 3
        
        pts3d1 = pts3d_world1.view(1, image_height, image_width, 3)
        pts3d2 = pts3d_world2.view(1, image_height, image_width, 3)

        conf1 = torch.ones(1, image_height, image_width).cuda() * 30
        conf2 = torch.ones(1, image_height, image_width).cuda() * 30

        return pts3d1, pts3d2, conf1, conf2

    def refine_pts(self, view1, view2, res_view1_dict, res_view2_dict, view_info1, view_info2):
        from plana3r.utils.merge_util_for_ref import merge_plane_fast_pcd, merge_plane
        from pytorch3d.ops import knn_points
        from planar_splatting.utils.mesh_utils import render_depth
        import open3d as o3d
        import trimesh
        image_height, image_width = view1['img'].shape[-2:]


        combined_plane_normal = torch.cat([res_view1_dict['plane_normal_list'][0], res_view2_dict['plane_normal_list'][0]], dim=0)
        combined_plane_center = torch.cat([res_view1_dict['plane_center_local_list'][0], res_view2_dict['plane_center_local_list'][0]], dim=0)
        combined_plane_radii = torch.cat([res_view1_dict['plane_radii_list'][0], res_view2_dict['plane_radii_list'][0]], dim=0)
        combined_plane_rot_q = torch.cat([res_view1_dict['plane_rot_q_normed_list'][0], res_view2_dict['plane_rot_q_normed_list'][0]], dim=0)
        
        planar_mesh, plane_ins_id_new = merge_plane(
                combined_plane_normal, 
                combined_plane_center, 
                combined_plane_radii[:,:2],  
                combined_plane_rot_q, 
                coarse_mesh_o3d=None,
                mesh_dist_thresh=0.02,
                plane_ins_id=None, 
                normal_angle_thresh=25,
                dist_thresh=0.1, 
                floor_height=-1,
                ceiling_height=-1,
                space_resolution=0.05,
                voxel_size=0.02,
                return_ins_parameters=False,
                colorMap_vis=None
            )

        o3d.io.write_triangle_mesh('output_mesh.ply',planar_mesh)
        planar_mesh = trimesh.load_mesh('output_mesh.ply')

        c2w1 = [res_view1_dict['pred_c2w_list'][0].cpu().numpy()]
        intrinsics1 = [view1['camera_intrinsics'][0].cpu().numpy()]
        depth1 = render_depth(planar_mesh, c2w1, intrinsics1, image_height, image_width)[0]
        rast_depth1 = torch.from_numpy(depth1).cuda().float()

        c2w2 = [res_view2_dict['pred_c2w_list'][0].cpu().numpy()]
        intrinsics2 = [view2['camera_intrinsics'][0].cpu().numpy()]
        depth2 = render_depth(planar_mesh, c2w2, intrinsics2, image_height, image_width)[0]
        rast_depth2 = torch.from_numpy(depth2).cuda().float()

        # rast_depth1 = res_view1_dict['allmaps_list'][0][0] # h, w
        # rast_depth2 = res_view2_dict['allmaps_list'][0][0] # h, w
        image_height, image_width = rast_depth1.shape

        if (self.uv is None) or (self.uv.shape[0] != image_height*image_width):
            uv = np.mgrid[0:image_height, 0:image_width].astype(np.int32)
            uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float().cuda()
            uv = uv.reshape(2, -1).transpose(1, 0)  # h*w, 2            
            self.uv = uv
        else:
            uv =self.uv
        uv1 = torch.cat([uv, torch.ones_like(uv[:, 0:1])], dim=-1)  # hw, 3

        intrinsic1 = view1['camera_intrinsics'][0]
        intrinsic1_inv = torch.inverse(intrinsic1)  # 3, 3
        pts3d_local1 = ((intrinsic1_inv @ uv1.t()) * rast_depth1.reshape(1, -1)).transpose(1, 0)  # hw, 3   
        pts3d_world1 = pts3d_local1

        c2w = res_view2_dict['pred_c2w_list'][0] # 4,4
        intrinsic2 = view2['camera_intrinsics'][0]
        intrinsic2_inv = torch.inverse(intrinsic2)  # 3, 3
        pts3d_local2 = ((intrinsic2_inv @ uv1.t()) * rast_depth2.reshape(1, -1)).transpose(1, 0)  # hw, 3   
        pts3d_local_homo2 = torch.cat([pts3d_local2, torch.ones_like(pts3d_local2[:,0:1])], dim=-1)
        pts3d_world2 = torch.matmul(c2w.cuda(), pts3d_local_homo2.t())
        pts3d_world2 = pts3d_world2.t()[:, :3] # npts, 3
        
        res_view1_dict['pts3d'] = pts3d_world1.view(1, image_height, image_width, 3)
        res_view2_dict['pts3d_in_other_view'] = pts3d_world2.view(1, image_height, image_width, 3)

    def forward(self, view1, view2, warmup=False, include_gt_geo=True, training=True, use_pred_intrinsic=False):
        (shape1, shape2), (feat1, feat2), (pos1, pos2) = self.encoder._encode_symmetrized(view1, view2)
        bs = feat1.shape[0]
        # feat1: B, S, C_enc
        # pos1: B, S, 2
        dec1, dec2 = self.encoder._decoder(feat1, pos1, feat2, pos2)
        
        true_shape1 = view1['true_shape']
        true_shape2 = view2['true_shape']
        assert shape1[0][0] == shape2[0][0] and shape1[0][1] == shape2[0][1]
        assert true_shape1[0][0] == shape1[0][0] and true_shape1[0][1] == shape1[0][1]
        assert true_shape2[0][0] == shape2[0][0] and true_shape2[0][1] == shape2[0][1]

        height = shape1[0][0]
        width = shape1[0][1]

        dec1_last = dec1[-1] # (2, S, dec_dim)
        dec2_last = dec2[-1]
        dec1_last_upsampled = self.upsample_head(dec1_last, shape1[0], self.patch_size) # (2, 4S, dec_dim)
        dec2_last_upsampled = self.upsample_head(dec2_last, shape2[0], self.patch_size) # (2, 4s, dec_dim)

        # predict camera pose
        if self.pose_type == 'simple':
            dec_pair = torch.cat([dec1_last, dec2_last], dim = 1)
        else:
            dec_pair = torch.cat([dec1_last_upsampled, dec2_last_upsampled], dim = 1)
        
        pred_rel_rot_q_normed, pred_rel_rot_q, pred_rel_trans = self.camera_head(dec_pair)

        # predict fov
        pred_fov1 = self.camera_head_fov(dec1_last)
        pred_fov2 = self.camera_head_fov(dec2_last)

        # convert to intrinsic matrix
        if use_pred_intrinsic:
            print('using pred intrinsics......')
            pred_intrinsic_matrix1 = fov_to_intrinsic_matrix(pred_fov1, height, width)  # b, 3, 3
            pred_intrinsic_matrix2 = fov_to_intrinsic_matrix(pred_fov2, height, width)
            # assert 'camera_intrinsics' in view1.keys() and 'camera_intrinsics' in view2.keys()
            view1['camera_intrinsics'] = pred_intrinsic_matrix1
            view2['camera_intrinsics'] = pred_intrinsic_matrix2

        # predict points map
        pred_pts1_dict = self.points3d_head(dec1_last, 'view1', shape1[0])
        pred_pts2_dict = self.points3d_head(dec2_last, 'view2', shape2[0])

        # predict planes parameters (center_depth, radii, rot_q)
        pred_view1_low = self.plane_head(dec1_last, 'view1')
        pred_view2_low = self.plane_head(dec2_last, 'view2')
        pred_view1_high = self.plane_head(dec1_last_upsampled, 'view1')
        pred_view2_high = self.plane_head(dec2_last_upsampled, 'view2')

        # get rel pose
        rel_rot_matrix = model_util.quat_to_rot(pred_rel_rot_q_normed)  # bs, 3, 3
        c2w_pred1 = torch.eye(4).cuda().unsqueeze(0).repeat(bs, 1, 1)  # bs, 4, 4
        c2w_pred2 = torch.eye(4).cuda().unsqueeze(0).repeat(bs, 1, 1)  # bs, 4, 4
        c2w_pred2[:, :3,:3] = rel_rot_matrix[:,:3,:3]
        c2w_pred2[:, :3, 3] = pred_rel_trans

        # get gt pose
        c2w_gt1 = torch.eye(4).cuda().unsqueeze(0).repeat(bs, 1, 1)  # bs, 4, 4
        c2w_gt2 = torch.bmm(torch.linalg.inv(view1['c2w']), view2['c2w'])

        # get plane prim (low & high) of each view
        pred_prim1_low = self.get_primitives(pred_view1_low, view1, 'view1', c2w_gt1 if training else c2w_pred1, warm_up=warmup)
        pred_prim1_high = self.get_primitives(pred_view1_high, view1, 'view1', c2w_gt1 if training else c2w_pred1, warm_up=warmup)
        pred_prim2_low = self.get_primitives(pred_view2_low, view2, 'view2', c2w_gt2 if training else c2w_pred2, warm_up=warmup)
        pred_prim2_high = self.get_primitives(pred_view2_high, view2, 'view2', c2w_gt2 if training else c2w_pred2, warm_up=warmup)
        
        # get view info
        viewinfo_list1 = self.get_view_info(view1, c2w_gt1 if training else c2w_pred1, include_gt_geo=include_gt_geo)
        viewinfo_list2 = self.get_view_info(view2, c2w_gt2 if training else c2w_pred2, include_gt_geo=include_gt_geo)

        # render plane prim
        res_view1_dict = self.render_primitives(view1, viewinfo_list1, pred_prim1_low, pred_prim1_high, training=training, warmup=warmup)
        res_view2_dict = self.render_primitives(view2, viewinfo_list2, pred_prim2_low, pred_prim2_high, training=training, warmup=warmup)

        # add view info list
        res_view1_dict['viewinfo_list'] = viewinfo_list1
        res_view2_dict['viewinfo_list'] = viewinfo_list2

        # add pred c2w (rel pose)
        res_view1_dict['pred_c2w_list'] = [c2w_b for c2w_b in c2w_pred1]
        res_view2_dict['pred_c2w_list'] = [c2w_b for c2w_b in c2w_pred2]

        # add points map
        res_view1_dict.update(pred_pts1_dict)
        res_view2_dict.update(pred_pts2_dict)
        res_view2_dict['pts3d_in_other_view'] = res_view2_dict.pop('pts3d')  # predict view2's pts3d in view1's frame

        # add fov
        res_view1_dict['fov'] = pred_fov1
        res_view2_dict['fov'] = pred_fov2

        return res_view1_dict, res_view2_dict, pred_rel_trans, pred_rel_rot_q
        
    def infer_planes(self, view1, view2, include_gt_geo=False, use_pred_intrinsic=False):
        assert not self.training
        res_view1_dict, res_view2_dict, pred_rel_trans, pred_rel_rot_q = self.forward(view1, view2, warmup=False, include_gt_geo=include_gt_geo, training=False, use_pred_intrinsic=use_pred_intrinsic)
        return res_view1_dict, res_view2_dict
          
    def draw_plane(self, plane_normal, plane_center, plane_radii,  plane_rot_q, suffix='initial-mono-cues', epoch=-1, to_unscaled_coord=False, plane_id=None):
        mesh = plot_util.plot_rectangle_planes(
            plane_center, plane_normal, plane_radii, plane_rot_q, 
            epoch=epoch, 
            suffix='%s'%(suffix), 
            to_unscaled_coord=to_unscaled_coord, 
            pose_cfg=None, 
            out_path=self.plot_dir,
            plane_id=plane_id, 
            color_type='normal')
        mesh = plot_util.plot_rectangle_planes(
            plane_center, plane_normal, plane_radii, plane_rot_q, 
            epoch=epoch, 
            suffix='%s'%(suffix), 
            to_unscaled_coord=to_unscaled_coord, 
            pose_cfg=None, 
            out_path=self.plot_dir,
            plane_id=plane_id, 
            color_type='prim')
        return mesh
