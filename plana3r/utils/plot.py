
import numpy as np
from PIL import Image

def save_normal_tensor_as_png(normal_tensor, output_path):
    if normal_tensor.shape[0] == 768:
        h = 24
        w = 32
    elif normal_tensor.shape[0] == 3072:
        h = 48
        w = 64
    normal_grid = normal_tensor.view(h, w, 3)
    try:
        normal_np = normal_grid.detach().cpu().numpy()
    except:
        normal_np = normal_grid.detach().cpu().numpy()
    
    normal_uint8 = np.clip((normal_np + 1) * 127.5, 0, 255).astype(np.uint8)
    
    Image.fromarray(normal_uint8, 'RGB').save(output_path)

def save_depth_tensor_as_png(depth_tensor, output_path):
    if depth_tensor.shape[0] == 768:
        h = 24
        w = 32
    elif depth_tensor.shape[0] == 3072:
        h = 48
        w = 64
    
    depth_2d = depth_tensor.view(h, w)
    
    try:
        depth_np = depth_2d.cpu().numpy()
    except:
        depth_np = depth_2d.detach().cpu().numpy()
    
    # norm to [0, 255]
    min_val, max_val = depth_np.min(), depth_np.max()
    if max_val - min_val == 0:
        depth_normalized = np.zeros_like(depth_np, dtype=np.uint8)  
    else:
        depth_normalized = ((depth_np - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    
    Image.fromarray(depth_normalized, mode='L').save(output_path)
    #print(f"save img to: {output_path}")
