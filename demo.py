#!/usr/bin/env python3
"""Gradio demo launcher for Plana3r multi-view planar reconstruction."""

from __future__ import annotations
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import gradio as gr
import numpy as np
# Plotly no longer used for display; keep imports removed to avoid extra deps
import open3d as o3d
import torch
import torch.nn.functional as F
from torch import nn

from plana3r.inference_plana3r import inference_plana3r
from plana3r.model_plana3r_v2 import Plana3rModel as Plana3rV2
from plana3r.model_plana3r_v1 import Plana3rModelNaive as Plana3rV1
from plana3r.utils.image import load_images_cameras
from plana3r.utils.inference import (
    get_coordinate_map,
    get_per_view_rast_segmap_segparam,
    get_random_color_map,
    labelcolormap,
    merge_primitives,
    rast_primitives,
    upadte_plane_parameters,
    writePCDFile,
    writePLYFile,
)
from plana3r.utils.misc import get_raster_cameras_simple
from planar_splatting.utils.model_util import quaternion_mult, rot_to_quat


torch.backends.cudnn.benchmark = True

# Planar inference expects square-resized images at 512px by default.
IMAGE_RESOLUTION = 512
MAX_PLOT_POINTS = 50000
GALLERY_TILE_HEIGHT = 400  # px per image in sidebar galleries
# Inflate exported GLB scene scale (viewer units). >1 makes it look larger
VIEWER_SCALE_MULTIPLIER = 1.5
BLOCKS_CSS = """
#rgb-view, #seg-view {
  width: 100%;
  height: 600px; /* Align with two 300px side sections */
  background: #ffffff;
  border-radius: 1px;
  box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
}

/* Make the Upload images button red */
#upload-btn button {
  background-color: #e53935 !important; /* red */
  border-color: #e53935 !important;
  color: #000 !important; /* black text */
}
#upload-btn button:hover {
  background-color: #ff5252 !important; /* lighter red on hover */
  border-color: #ff5252 !important;
  color: #000 !important;
}

/* Make the Run reconstruction button green */
#run-btn button {
  background-color: #2e7d32 !important;  /* green 800 */
  border-color: #2e7d32 !important;
  color: #fff !important;
}
#run-btn button:hover {
  background-color: #1b5e20 !important;  /* green 900 */
  border-color: #1b5e20 !important;
}

#rgb-view canvas, #seg-view canvas {
  height: 100% !important;
}

.gradio-container {
  max-width: 1152px !important;
}
"""

# Default checkpoints shipped with the release.
DEFAULT_CHECKPOINTS = {
    "v1": Path("checkpoints/plana3r_v1_official.pth"),
    "v2": Path("checkpoints/plana3r_v2_official.pth"),
}

# Cache to avoid reloading checkpoints between Gradio runs.
_MODEL_CACHE: Dict[Tuple[str, str, str], nn.Module] = {}


def _resolve_upload_path(upload) -> Path:
    """Return a filesystem Path for a Gradio upload object."""
    for attr in ("name", "path"):
        candidate = getattr(upload, attr, None)
        if candidate:
            path = Path(candidate)
            if path.exists():
                return path
    raise ValueError("Unable to resolve uploaded file path.")


def make_pairs_simple(sequence: Sequence) -> List[Tuple]:
    """Create adjacent pairs from a sequence of view dicts."""
    return [tuple(sequence[i : i + 2]) for i in range(len(sequence) - 1)]


def load_plana3r_model(
    version: str,
    checkpoint_path: Path,
    pose_type: str,
    device: torch.device,
) -> nn.Module:
    """Load (and cache) the requested Plana3r checkpoint."""
    cache_key = (version, str(checkpoint_path), pose_type)
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        if version == "v2":
            model = Plana3rV2(
                pretrained_model_name_or_path="",
                pos_embed="RoPE100",
                img_size=(IMAGE_RESOLUTION, IMAGE_RESOLUTION),
                head_type="linear",
                output_mode="pts3d",
                depth_mode=("linear", -100, 100),
                enc_embed_dim=1024,
                enc_depth=24,
                enc_num_heads=16,
                dec_embed_dim=768,
                dec_depth=12,
                dec_num_heads=12,
                pose_type=pose_type,
            )
        elif version == "v1":
            model = Plana3rV1(
                pretrained_model_name_or_path="",
                pos_embed="RoPE100",
                img_size=(IMAGE_RESOLUTION, IMAGE_RESOLUTION),
                head_type="linear",
                output_mode="pts3d",
                depth_mode=("linear", -100, 100),
                pose_type=pose_type,
            )
        else:
            raise ValueError(f"Unknown model version '{version}'.")

        state = torch.load(checkpoint_path, map_location=device)
        state = state["model"] if isinstance(state, dict) and "model" in state else state
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        for param in model.parameters():
            param.requires_grad_(False)
        _MODEL_CACHE[cache_key] = model
    else:
        model.to(device).eval()
    return model


def prepare_intrinsics(
    intrinsic_path: Optional[Path],
    num_views: int,
    version: str,
) -> List[np.ndarray]:
    """Return a list of intrinsics (one per input view)."""
    if intrinsic_path is None:
        if version == "v1":
            raise ValueError("Intrinsics are required when using the v1 model.")
        intrinsic = np.eye(3, dtype=np.float32)
    else:
        intrinsic = np.loadtxt(str(intrinsic_path)).astype(np.float32)[:3, :3]
    return [intrinsic] * num_views


def blend_segmentation(segmentation: np.ndarray, image_bgr: np.ndarray) -> np.ndarray:
    """Return an RGB overlay of planar segmentation on the original view."""
    colors = labelcolormap(256)
    seg_colors = colors[segmentation]
    blended = (seg_colors * 0.7 + image_bgr.astype(np.float32) * 0.3).astype(np.uint8)
    mask = (segmentation > 0).astype(np.uint8)[..., None]
    blended = blended * mask + image_bgr.astype(np.uint8) * (1 - mask)
    return cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)


def fuse_point_clouds(
    depth_list: Sequence[np.ndarray],
    segmentation_list: Sequence[np.ndarray],
    image_list: Sequence[np.ndarray],
    coord_map_list: Sequence[np.ndarray],
    out_h: int,
    out_w: int,
    c2w_list: Sequence[np.ndarray],
    color_map_vis: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Fuse per-view reconstructions into shared point clouds."""
    rgb_points: List[np.ndarray] = []
    rgb_colors: List[np.ndarray] = []
    seg_colors: List[np.ndarray] = []

    for depth, segmentation, image, coord_map, c2w in zip(
        depth_list, segmentation_list, image_list, coord_map_list, c2w_list
    ):
        depth_resized = cv2.resize(depth, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        image_resized = cv2.resize(image, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        segmentation_resized = cv2.resize(
            segmentation, (out_w, out_h), interpolation=cv2.INTER_NEAREST
        ).astype(np.int32)

        mask = segmentation_resized > 0
        if not np.any(mask):
            continue

        coord_flat = coord_map.reshape(3, -1)
        depth_flat = depth_resized.reshape(1, -1)
        pts_local = (coord_flat * depth_flat).transpose(1, 0)

        if c2w is not None:
            pts_local_h = np.concatenate(
                [pts_local, np.ones((pts_local.shape[0], 1), dtype=pts_local.dtype)], axis=1
            )
            pts_world = (c2w @ pts_local_h.T).T[:, :3]
        else:
            pts_world = pts_local

        mask_flat = mask.reshape(-1)
        pts_world = pts_world[mask_flat]
        colors_rgb = image_resized.reshape(-1, 3)[mask_flat] / 255.0
        seg_ids = segmentation_resized.reshape(-1)[mask_flat]
        colors_seg = color_map_vis[seg_ids] / 255.0

        rgb_points.append(pts_world.astype(np.float32))
        rgb_colors.append(colors_rgb.astype(np.float32))
        seg_colors.append(colors_seg.astype(np.float32))

    if not rgb_points:
        return None, None, None

    fused_points = np.concatenate(rgb_points, axis=0)
    fused_rgb = np.concatenate(rgb_colors, axis=0)
    fused_seg = np.concatenate(seg_colors, axis=0)
    is_nan = np.isnan(fused_points).sum(axis=1)>0

    return fused_points[~is_nan], fused_rgb[~is_nan], fused_seg[~is_nan]


# Plotly helpers removed; using GLB + Model3D for visualization


def ply_to_glb(
    ply_path: Path,
    transform: Optional[np.ndarray] = None,
    extras: Optional[Sequence["trimesh.Trimesh"]] = None,
) -> Optional[Path]:
    """Convert a PLY mesh to GLB using trimesh, with optional 4x4 transform, extra meshes, and sanitization."""
    try:
        import trimesh

        obj = trimesh.load(str(ply_path), process=False)
        # Flatten scenes to a single mesh if needed
        if isinstance(obj, trimesh.Scene):
            if len(obj.geometry) == 0:
                return None
            mesh = trimesh.util.concatenate(tuple(obj.geometry.values()))
        else:
            mesh = obj

        # Ensure faces exist (point clouds can't be exported as GLB reliably)
        if getattr(mesh, 'faces', None) is None or len(mesh.faces) == 0:
            return None

        # Optional reorientation/normalization transform
        if transform is not None:
            try:
                mesh.apply_transform(transform)
            except Exception:
                pass

        # Sanitize vertices/faces: remove faces that touch invalid vertices
        v = mesh.vertices
        f = mesh.faces
        if not isinstance(v, np.ndarray) or not isinstance(f, np.ndarray) or v.size == 0 or f.size == 0:
            return None
        finite_v = np.isfinite(v).all(axis=1)
        # Also reject extreme coordinates
        finite_v &= (np.abs(v) < 1e6).all(axis=1)
        face_ok = finite_v[f].all(axis=1)
        if not face_ok.any():
            return None
        # Submesh with valid faces (reindexes vertices automatically)
        sub = mesh.submesh([np.flatnonzero(face_ok)], append=True, repair=True)
        if isinstance(sub, list):
            sub = sub[0] if len(sub) > 0 else None
        if sub is None or len(sub.faces) == 0 or len(sub.vertices) == 0:
            return None
        # Optional cleanup
        try:
            sub.remove_degenerate_faces()
            sub.remove_unreferenced_vertices()
        except Exception:
            pass

        # Merge with any extra meshes (e.g., camera frustums)
        if extras:
            try:
                # Ensure extras are transformed consistently
                meshes = [sub]
                for m in extras:
                    mm = m.copy()
                    if transform is not None:
                        try:
                            mm.apply_transform(transform)
                        except Exception:
                            pass
                    meshes.append(mm)
                sub = trimesh.util.concatenate(meshes)
            except Exception:
                pass
        
        v = sub.vertices
        v[:, 2] *= -1
        sub.vertices = v

        material = trimesh.visual.material.PBRMaterial(doubleSided=True)
        sub.visual.material = material

        glb_bytes = trimesh.exchange.gltf.export_glb(sub)
        out = ply_path.with_suffix('.glb')
        with open(out, 'wb') as f:
            f.write(glb_bytes)
        return out
    except Exception:
        return None


def compute_viewer_transform(c2w_list: Sequence[np.ndarray], pts: Optional[np.ndarray]) -> np.ndarray:
    """Compute a transform that recenters, rescales, and aligns world to glTF (Y-up, -Z forward)."""
    is_nan = np.isnan(pts).sum(axis=1)>0
    pts = pts[~is_nan]
    # Average camera axes in world
    ups = []
    forwards = []
    for c2w in c2w_list:
        R = c2w[:3, :3]
        cam_y = R[:, 1]  # camera y in world (down in OpenCV)
        cam_z = R[:, 2]  # camera z in world (forward)
        ups.append(-cam_y)  # up is negative camera y
        forwards.append(cam_z)
    up_avg = np.mean(np.stack(ups, axis=0), axis=0)
    fwd_avg = np.mean(np.stack(forwards, axis=0), axis=0)
    # Orthonormalize basis
    def _norm(x):
        n = np.linalg.norm(x) + 1e-9
        return x / n
    s_up = _norm(up_avg)
    # Make forward orthogonal to up
    fwd_proj = fwd_avg - np.dot(fwd_avg, s_up) * s_up
    s_fwd = _norm(fwd_proj if np.linalg.norm(fwd_proj) > 1e-9 else fwd_avg)
    s_right = _norm(np.cross(s_up, s_fwd))
    s_fwd = _norm(np.cross(s_right, s_up))
    S = np.stack([s_right, s_up, s_fwd], axis=1)  # columns
    # Target glTF basis: right=[1,0,0], up=[0,1,0], forward=[0,0,-1]
    t_right = np.array([1.0, 0.0, 0.0])
    t_up = np.array([0.0, 1.0, 0.0])
    t_fwd = np.array([0.0, 0.0, -1.0])
    T = np.stack([t_right, t_up, t_fwd], axis=1)
    R_align = T @ S.T  # map source basis to target
    # Center and scale
    if pts is not None and pts.size > 0:
        center = np.median(pts, axis=0)
        radii = np.linalg.norm(pts - center, axis=1)
        scale = np.percentile(radii, 95)
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
    else:
        center = np.zeros(3)
        scale = 1.0
    s = VIEWER_SCALE_MULTIPLIER / scale
    M = np.eye(4)
    M[:3, :3] = s * R_align
    M[:3, 3] = -(s * (R_align @ center))
    return M


def build_camera_frusta_edge_tubes(
    c2w_list: Sequence[np.ndarray],
    intrinsics_list: Sequence[np.ndarray],
    width: int,
    height: int,
    near: float,
    far: float,
    radius: float,
    color_rgb: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    sections: int = 8,
) -> List["trimesh.Trimesh"]:
    import trimesh

    def cam_corners(K: np.ndarray, w: int, h: int, z: float) -> np.ndarray:
        uv = np.array([
            [0, 0, 1.0],
            [w, 0, 1.0],
            [w, h, 1.0],
            [0, h, 1.0],
        ], dtype=np.float64)
        Kinv = np.linalg.inv(K)
        rays = (Kinv @ uv.T).T  # 4x3
        return rays * z

    def tube_between(p0: np.ndarray, p1: np.ndarray) -> Optional["trimesh.Trimesh"]:
        v = p1 - p0
        L = np.linalg.norm(v)
        if not np.isfinite(L) or L < 1e-8:
            return None
        axis = v / L
        cyl = trimesh.creation.cylinder(radius=radius, height=L, sections=sections)
        # cylinder centered at origin along z in [-L/2, L/2]; align +Z to axis
        R = trimesh.geometry.align_vectors(np.array([0.0, 0.0, 1.0]), axis)
        cyl.apply_transform(R)
        cyl.apply_translation((p0 + p1) / 2.0)
        return cyl

    color255 = np.array(color_rgb, dtype=np.float64)
    meshes: List[trimesh.Trimesh] = []
    for c2w, K in zip(c2w_list, intrinsics_list):
        O_cam = np.zeros((1, 3), dtype=np.float64)
        near_c = cam_corners(K, width, height, near)
        # Define edges: 4 rays and 4 near-rectangle edges
        edges_cam = []
        # Rays O->near corners
        for i in range(4):
            edges_cam.append((O_cam[0], near_c[i]))
        # Near rectangle
        for i in range(4):
            edges_cam.append((near_c[i], near_c[(i + 1) % 4]))

        # Transform to world and create tubes
        for a_c, b_c in edges_cam:
            ah = np.append(a_c, 1.0)
            bh = np.append(b_c, 1.0)
            aw = (c2w @ ah)[:3]
            bw = (c2w @ bh)[:3]
            tube = tube_between(aw, bw)
            if tube is None:
                continue
            vc = np.tile((color255 * 255.0).clip(0, 255), (tube.vertices.shape[0], 1))
            tube.visual.vertex_colors = vc
            meshes.append(tube)
    return meshes

def run_plana3r_demo(
    image_files: Sequence,
    output_mode: str,
):
    """Gradio callback: execute inference and prepare outputs."""
    if not torch.cuda.is_available():
        raise gr.Error("CUDA device not found. This demo currently requires a GPU.")
    if not image_files or len(image_files) < 2:
        raise gr.Error("Upload at least two images for multi-view inference.")

    device = torch.device("cuda")
    # Use v2 official checkpoint by default
    checkpoint = Path(DEFAULT_CHECKPOINTS["v2"]).expanduser()

    if not checkpoint.exists():
        raise gr.Error(
            f"Checkpoint not found: {checkpoint} "
            "(supply a custom path or ensure the official checkpoints are present)."
        )

    img_paths = sorted((_resolve_upload_path(f) for f in image_files), key=lambda p: p.name)
    img_raw_list = []
    for path in img_paths:
        img = cv2.imread(str(path))
        if img is None:
            raise gr.Error(f"Failed to read image: {path}")
        img_raw_list.append(img)

    intrinsic_list = prepare_intrinsics(None, len(img_paths), "v2")
    c2w_identity_list = [np.eye(4, dtype=np.float32)] * len(img_paths)

    images_list = load_images_cameras(
        [str(p) for p in img_paths],
        size=IMAGE_RESOLUTION,
        camera_intrinsics_list=intrinsic_list,
        camera_extrinsics_list=c2w_identity_list,
    )
    all_pairs = make_pairs_simple(images_list)

    model = load_plana3r_model("v2", checkpoint, "simple", device)
    use_pred_intrinsic = True

    with torch.no_grad():
        output_view1, output_view2 = inference_plana3r(
            all_pairs,
            model,
            device,
            batch_size=1,
            include_gt_geo=False,
            use_pred_intrinsic=use_pred_intrinsic,
        )

    c2ref_pose = torch.eye(4, device=device)
    plane_center_world_list: List[torch.Tensor] = []
    plane_radii_world_list: List[torch.Tensor] = []
    plane_normal_world_list: List[torch.Tensor] = []
    plane_rot_q_world_list: List[torch.Tensor] = []
    intrinsic_raw_list: List[torch.Tensor] = []
    c2w_seq_list: List[torch.Tensor] = []

    height_raw, width_raw = img_raw_list[0].shape[:2]
    resize_scale_h = height_raw / images_list[0]["img"].shape[-2]
    resize_scale_w = width_raw / images_list[0]["img"].shape[-1]

    for idx in range(len(all_pairs)):
        intrinsic_resized_v1 = output_view1["viewinfo_list"][idx].intrinsic
        intrinsic_resized_v2 = output_view2["viewinfo_list"][idx].intrinsic
        intrinsic_raw_v1 = intrinsic_resized_v1.clone()
        intrinsic_raw_v2 = intrinsic_resized_v2.clone()
        intrinsic_raw_v1[0] *= resize_scale_w
        intrinsic_raw_v1[1] *= resize_scale_h
        intrinsic_raw_v2[0] *= resize_scale_w
        intrinsic_raw_v2[1] *= resize_scale_h

        if not intrinsic_raw_list:
            intrinsic_raw_list = [intrinsic_raw_v1, intrinsic_raw_v2]
        else:
            intrinsic_raw_list.append(intrinsic_raw_v2)

        plane_center = torch.cat(
            [
                output_view1["plane_center_local_list"][idx],
                output_view2["plane_center_local_list"][idx],
            ],
            dim=0,
        ).to(device)
        plane_radii = torch.cat(
            [
                output_view1["plane_radii_list"][idx],
                output_view2["plane_radii_list"][idx],
            ],
            dim=0,
        ).to(device)
        plane_normal = torch.cat(
            [
                output_view1["plane_normal_list"][idx],
                output_view2["plane_normal_list"][idx],
            ],
            dim=0,
        ).to(device)
        plane_rot_q = torch.cat(
            [
                output_view1["plane_rot_q_normed_list"][idx],
                output_view2["plane_rot_q_normed_list"][idx],
            ],
            dim=0,
        ).to(device)

        plane_center_homo = torch.cat(
            [plane_center, torch.ones((plane_center.shape[0], 1), device=device)],
            dim=-1,
        )
        plane_center_world = (c2ref_pose @ plane_center_homo.t()).t()[..., :3]
        c2ref_pose_quat = rot_to_quat(c2ref_pose[None])
        plane_rot_q_world = quaternion_mult(c2ref_pose_quat, plane_rot_q)
        plane_normal_world = (c2ref_pose[:3, :3] @ plane_normal.t()).t()

        rel_pose = output_view2["pred_c2w_list"][idx].to(device)
        c2ref_pose = c2ref_pose @ rel_pose
        if not c2w_seq_list:
            c2w_seq_list = [
                output_view1["pred_c2w_list"][idx].to(device),
                output_view2["pred_c2w_list"][idx].to(device),
            ]
        else:
            c2w_seq_list.append(c2ref_pose)

        plane_center_world_list.append(plane_center_world)
        plane_radii_world_list.append(plane_radii)
        plane_normal_world_list.append(plane_normal_world)
        plane_rot_q_world_list.append(plane_rot_q_world)

    plane_ins_ids, valid_mask = merge_primitives(
        plane_normal_world_list,
        plane_center_world_list,
        plane_radii_world_list,
        plane_rot_q_world_list,
        merge_version="stable",
    )
    max_id = plane_ins_ids.max()
    if max_id.item() <= 0:
        raise gr.Error("No planar primitives were detected.")
    plane_ins_ids_norm = plane_ins_ids.float() / max_id
    plane_ins_ids_norm_c3 = plane_ins_ids_norm[..., None].repeat(1, 3)

    color_map_vis = get_random_color_map(12000)(10000)
    num_views = len(c2w_seq_list)
    view_info_list = [
        get_raster_cameras_simple(
            intrinsic_raw_list[v].to(device),
            c2w_seq_list[v].to(device),
            height=height_raw,
            width=width_raw,
        )
        for v in range(num_views)
    ]

    rgb_ins_ids_list, allmap_list = rast_primitives(
        view_info_list,
        plane_center_world_list,
        plane_radii_world_list,
        plane_rot_q_world_list,
        plane_ins_ids_norm_c3,
        plane_ins_ids,
        height=height_raw,
        width=width_raw,
    )
    plane_center_updated, _, plane_rot_q_updated, _ = upadte_plane_parameters(
        plane_center_world_list,
        plane_normal_world_list,
        plane_rot_q_world_list,
        plane_radii_world_list,
        plane_ins_ids,
    )
    _, allmap_list = rast_primitives(
        view_info_list,
        plane_center_updated,
        plane_radii_world_list,
        plane_rot_q_updated,
        plane_ins_ids_norm_c3,
        plane_ins_ids,
        height=height_raw,
        width=width_raw,
    )

    seg_map_global_list: List[torch.Tensor] = []
    plane_depth_list: List[np.ndarray] = []
    pred_seg_list: List[np.ndarray] = []
    out_k_inv_dot_xy_1_list: List[np.ndarray] = []
    gallery_entries: List[Tuple[np.ndarray, str]] = []

    plot_h, plot_w = 192, 256

    for view_idx in range(num_views):
        seg_map_global, seg_map_local, seg_params, seg_masks = get_per_view_rast_segmap_segparam(
            allmap_list[view_idx],
            view_info_list[view_idx],
            rgb_ins_ids_list[view_idx],
            plane_ins_ids,
            min_mask_size=2500,
        )
        seg_map_global_list.append(seg_map_global)

        if not seg_masks:
            plane_depth = torch.zeros((height_raw, width_raw), device=device)
            pred_seg = torch.zeros((height_raw, width_raw), dtype=torch.int32, device=device)
        else:
            pred_masks = torch.stack(seg_masks, dim=0).float()
            pred_param = torch.stack(seg_params, dim=0)
            pred_offset = pred_param.norm(dim=-1, keepdim=True)
            pred_normal = F.normalize(pred_param, dim=-1)
            uv1 = torch.stack(
                [
                    torch.arange(0, width_raw).reshape(1, -1).repeat(height_raw, 1),
                    torch.arange(0, height_raw).reshape(-1, 1).repeat(1, width_raw),
                    torch.ones((height_raw, width_raw)),
                ],
                dim=-1,
            ).reshape(-1, 3).float().to(device)
            k_inv = torch.inverse(intrinsic_raw_list[view_idx]).float().to(device)
            plane_depth = torch.zeros(height_raw, width_raw, device=device)
            for offset, normal, seg_mask in zip(pred_offset, pred_normal, pred_masks):
                ray = k_inv @ uv1.t()
                depth_i = offset / torch.mm(normal.reshape(1, 3), ray)
                depth_i = depth_i.reshape(height_raw, width_raw)
                plane_depth = plane_depth + depth_i * (seg_mask > 0)
            pred_seg = torch.zeros_like(plane_depth, dtype=torch.int32)
            for mask_idx, seg_mask in enumerate(pred_masks):
                pred_seg[seg_mask > 0] = mask_idx + 1

        plane_depth_np = plane_depth.detach().cpu().numpy()
        pred_seg_np = pred_seg.detach().cpu().numpy().astype(np.int32)
        plane_depth_list.append(plane_depth_np)
        pred_seg_list.append(pred_seg_np)

        overlay = blend_segmentation(pred_seg_np, img_raw_list[view_idx])
        gallery_entries.append((overlay, f"View {view_idx}: {int(pred_seg_np.max())} planes"))

        coord_map = get_coordinate_map(
            intrinsic_raw_list[view_idx].detach().cpu().numpy(),
            height_raw,
            width_raw,
            plot_h,
            plot_w,
            device,
        )
        out_k_inv_dot_xy_1_list.append(coord_map.cpu().numpy().reshape(3, plot_h, plot_w))

    c2w_numpy_list = [pose.detach().cpu().numpy() for pose in c2w_seq_list]
    output_root = Path(tempfile.mkdtemp(prefix="plana3r_demo_"))
    suffix = f"planar_rgb_{num_views}views"
    seg_suffix = suffix.replace("rgb", "seg")
    rgb_model_path: Optional[Path] = None
    seg_model_path: Optional[Path] = None
    mesh_rgb_ply: Optional[Path] = None
    mesh_seg_ply: Optional[Path] = None
    if output_mode == "mesh":
        writePLYFile(
            str(output_root),
            suffix,
            plane_depth_list,
            pred_seg_list,
            img_raw_list,
            0,
            out_k_inv_dot_xy_1_list,
            plot_h,
            plot_w,
            c2w_numpy_list,
        )
        seg_map_color_list = []
        for seg_map_global in seg_map_global_list:
            seg_np = seg_map_global.detach().cpu().numpy()
            seg_color = color_map_vis[seg_np.reshape(-1)].reshape(
                seg_np.shape[0], seg_np.shape[1], 3
            ).astype(np.uint8)
            seg_map_color_list.append(seg_color)
        writePLYFile(
            str(output_root),
            seg_suffix,
            plane_depth_list,
            pred_seg_list,
            seg_map_color_list,
            0,
            out_k_inv_dot_xy_1_list,
            plot_h,
            plot_w,
            c2w_numpy_list,
        )
        mesh_rgb_ply = output_root / f"planar_{suffix}.ply"
        mesh_seg_ply = output_root / f"planar_{seg_suffix}.ply"
    else:
        writePCDFile(
            str(output_root),
            suffix,
            plane_depth_list,
            pred_seg_list,
            img_raw_list,
            0,
            out_k_inv_dot_xy_1_list,
            plot_h,
            plot_w,
            c2w_numpy_list,
        )
        seg_map_color_list = []
        for seg_map_global in seg_map_global_list:
            seg_np = seg_map_global.detach().cpu().numpy()
            seg_color = color_map_vis[seg_np.reshape(-1)].reshape(
                seg_np.shape[0], seg_np.shape[1], 3
            ).astype(np.uint8)
            seg_map_color_list.append(seg_color)
        writePCDFile(
            str(output_root),
            seg_suffix,
            plane_depth_list,
            pred_seg_list,
            seg_map_color_list,
            0,
            out_k_inv_dot_xy_1_list,
            plot_h,
            plot_w,
            c2w_numpy_list,
        )
        # Additionally, build mesh PLYs for web viewer using the same inputs
        writePLYFile(
            str(output_root),
            suffix,
            plane_depth_list,
            pred_seg_list,
            img_raw_list,
            0,
            out_k_inv_dot_xy_1_list,
            plot_h,
            plot_w,
            c2w_numpy_list,
        )
        writePLYFile(
            str(output_root),
            seg_suffix,
            plane_depth_list,
            pred_seg_list,
            seg_map_color_list,
            0,
            out_k_inv_dot_xy_1_list,
            plot_h,
            plot_w,
            c2w_numpy_list,
        )
        mesh_rgb_ply = output_root / f"planar_{suffix}.ply"
        mesh_seg_ply = output_root / f"planar_{seg_suffix}.ply"
    
    print(f"mesh_rgb_ply: {mesh_rgb_ply}")
    print(f"mesh_seg_ply: {mesh_seg_ply}")
    checkpoint_name = checkpoint.name
    fused_points, fused_rgb, fused_seg = fuse_point_clouds(
        plane_depth_list,
        pred_seg_list,
        img_raw_list,
        out_k_inv_dot_xy_1_list,
        plot_h,
        plot_w,
        c2w_numpy_list,
        color_map_vis.astype(np.float32),
    )
    
    # Compute viewer transform from camera poses and fused points
    viewer_T = compute_viewer_transform(c2w_numpy_list, fused_points)
    # Build camera frusta meshes (scaled by scene size)
    # Estimate scale from fused points to pick reasonable near/far
    if fused_points is not None and fused_points.size > 0:
        center = np.median(fused_points, axis=0)
        radii = np.linalg.norm(fused_points - center, axis=1)
        scene_scale = np.percentile(radii, 95)
        # Make frusta a bit larger
        near_z = max(scene_scale * 0.08, 1e-3)
        far_z = max(scene_scale * 0.24, near_z + 1e-3)
    else:
        near_z, far_z = 0.1, 0.5
    intri_np = [intr.cpu().numpy() if hasattr(intr, 'cpu') else np.array(intr) for intr in intrinsic_raw_list]
    # Build edge-based frusta (thin tubes) to approximate line frustums
    # Slightly thicker tubes for visibility
    tube_radius = max(scene_scale, 1.0) * 0.006 if fused_points is not None and fused_points.size > 0 else 0.015
    frusta_meshes = build_camera_frusta_edge_tubes(
        c2w_numpy_list, intri_np, width_raw, height_raw, near=near_z, far=far_z, radius=tube_radius, color_rgb=(1.0, 0.0, 0.0)
    )
    # Convert mesh PLYs to GLB for Model3D including frustums
    rgb_glb = ply_to_glb(mesh_rgb_ply, transform=viewer_T, extras=frusta_meshes) if mesh_rgb_ply and mesh_rgb_ply.exists() else None
    seg_glb = ply_to_glb(mesh_seg_ply, transform=viewer_T, extras=frusta_meshes) if mesh_seg_ply and mesh_seg_ply.exists() else None

    plane_count = int(torch.unique(plane_ins_ids[valid_mask]).numel())
    summary_parts = [
        f"Processed {len(img_paths)} views",
        f"merged {plane_count} planar instances",
        f"weights: {checkpoint_name}",
    ]
    if rgb_glb and Path(rgb_glb).exists():
        summary_parts.append(f"RGB GLB: {Path(rgb_glb).name}")
    if seg_glb and Path(seg_glb).exists():
        summary_parts.append(f"Seg GLB: {Path(seg_glb).name}")
    summary = " · ".join(summary_parts)
    # Only return GLB for display (no fallback)
    rgb_return = str(rgb_glb) if rgb_glb and Path(rgb_glb).exists() else None
    seg_return = str(seg_glb) if seg_glb and Path(seg_glb).exists() else None
    # Default display is RGB GLB
    default_display = rgb_return
    return gallery_entries, rgb_return, seg_return, default_display


with gr.Blocks(title="PLANA3R", css=BLOCKS_CSS) as demo:
    gr.Markdown( """
        ## [NeurIPS 2025] PLANA3R: Zero-shot Metric Planar 3D Reconstruction via Feed-Forward Planar Splatting
        ### [Paper](https://openreview.net/forum?id=YTwRZP8mNO) | [Project Page](https://lck666666.github.io/plana3r/) | [Code](https://github.com/lck666666/plana3r)
        
        **Quick Start:** Upload ≥2 images (ordered by filename) and click *Run reconstruction* to get a **planar 3D reconstruction**.
        """
    )
    with gr.Row():
        # Sidebar with input images and overlays
        with gr.Column(scale=1, min_width=320):
            upload_btn = gr.UploadButton(
                label="Upload images",
                file_types=["image"],
                file_count="multiple",
                elem_id="upload-btn",
            )
            input_gallery = gr.Gallery(label="Input images", columns=3, height=300, allow_preview=True)
            gallery_output = gr.Gallery(
                label="Segmentation overlays",
                columns=3,
                height=300,
                allow_preview=True,
            )
        # Main content: fused reconstruction viewer
        with gr.Column(scale=2):
            run_button = gr.Button("Run reconstruction", variant="primary", elem_id="run-btn")
            model_display = gr.Model3D(label="Plana3r Reconstruction", elem_id="rgb-view")
            with gr.Row():
                btn_show_rgb = gr.Button("Show Colored Mesh", elem_id="show-rgb-btn")
                btn_show_seg = gr.Button("Show Segmented Mesh", elem_id="show-seg-btn")
                output_mode_input = gr.Radio([
                    "pts3d",
                    "mesh",
                ], value="pts3d", label="Output mode")

    # Hidden states to hold GLB paths
    glb_rgb_state = gr.State()
    glb_seg_state = gr.State()
    image_files_state = gr.State()

    def _list_uploaded_paths(files):
        if not files:
            return gr.update(value=[], height=GALLERY_TILE_HEIGHT)
        paths = []
        for f in files:
            p = None
            for attr in ("name", "path"):
                val = getattr(f, attr, None)
                if val:
                    p = str(val)
                    break
            if p:
                paths.append(p)
        # Sort by filename for stable ordering
        try:
            paths = sorted(paths, key=lambda s: Path(s).name)
        except Exception:
            paths = sorted(paths)
        # h = max(1, len(paths)) * GALLERY_TILE_HEIGHT
        h = GALLERY_TILE_HEIGHT
        return gr.update(value=paths, height=h)

    def _pick_rgb(rgb_path, seg_path):
        return rgb_path

    def _pick_seg(rgb_path, seg_path):
        return seg_path

    run_button.click(
        run_plana3r_demo,
        inputs=[image_files_state, output_mode_input],
        outputs=[
            gallery_output,
            glb_rgb_state,
            glb_seg_state,
            model_display,
        ],
    )

    def _on_upload(files):
        if not files:
            return [], gr.update(value=[], height=300)
        try:
            files = sorted(files, key=lambda f: Path(getattr(f, 'name', getattr(f, 'path', ''))).name)
        except Exception:
            pass
        paths = []
        for f in files:
            for attr in ("name", "path"):
                val = getattr(f, attr, None)
                if val:
                    paths.append(str(val))
                    break
        return files, gr.update(value=paths, height=300)

    upload_btn.upload(_on_upload, inputs=upload_btn, outputs=[image_files_state, input_gallery])
    btn_show_rgb.click(_pick_rgb, inputs=[glb_rgb_state, glb_seg_state], outputs=model_display)
    btn_show_seg.click(_pick_seg, inputs=[glb_rgb_state, glb_seg_state], outputs=model_display)


if __name__ == "__main__":
    demo.launch()
