import os
import trimesh
import numpy as np

# 根目录（修改为你的路径）
root_dir = "./"

# 扫描所有子文件夹
for folder_name in os.listdir(root_dir):
    folder_path = os.path.join(root_dir, folder_name)
    if not os.path.isdir(folder_path):
        continue  # 只处理文件夹

    # 查找当前文件夹下所有 .ply 文件
    for file_name in os.listdir(folder_path):
        if file_name.lower().endswith(".ply"):
            ply_path = os.path.join(folder_path, file_name)
            glb_path = os.path.splitext(ply_path)[0] + ".glb"

            print(f"Converting {ply_path} → {glb_path}")
            try:
                mesh = trimesh.load(ply_path)
                pts = np.asarray(mesh.vertices).copy()
                # pts_tmp = np.asarray(mesh.vertices).copy()
                # pts[:, 1] = pts_tmp[:, 2]
                # pts[:, 2] = pts_tmp[:, 1]
                pts[:, 1] = -pts[:, 1]
                pts[:, 0] = -pts[:, 0]
                mesh.vertices = pts
                material = trimesh.visual.material.PBRMaterial(doubleSided=True)
                mesh.visual.material = material
                mesh.export(glb_path)
            except Exception as e:
                print(f"❌ Failed to convert {ply_path}: {e}")
