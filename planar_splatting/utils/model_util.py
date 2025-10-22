import torch
import torch.nn.functional as F
import quaternion

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

def rot_to_quat(R):
    """
    将旋转矩阵转换为四元数 (batch 版本)
    :param R: 旋转矩阵，形状为 (batch_size, 3, 3)
    :return: 四元数，形状为 (batch_size, 4)
    """
    batch_size, _, _ = R.shape
    q = torch.zeros((batch_size, 4), dtype=R.dtype, device=R.device)

    # 提取旋转矩阵的元素
    R00, R01, R02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    R10, R11, R12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    R20, R21, R22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    # 计算四元数的第一个分量 q0
    trace = R00 + R11 + R22  # 迹 (trace) 是旋转矩阵对角线元素的和
    q0 = 0.5 * torch.sqrt(torch.clamp(1.0 + trace, min=1e-8))  # 防止数值过小导致 NaN
    q[:, 0] = q0

    # 计算其他分量 q1, q2, q3
    q1 = (R21 - R12) / (4 * q0 + 1e-8)  # 添加一个小的 epsilon 防止除零
    q2 = (R02 - R20) / (4 * q0 + 1e-8)
    q3 = (R10 - R01) / (4 * q0 + 1e-8)

    q[:, 1] = q1
    q[:, 2] = q2
    q[:, 3] = q3

    # 归一化四元数以确保其单位长度
    q = q / torch.norm(q, dim=1, keepdim=True)

    return q

def get_rotation_quaternion_of_normal(plane_normal_init, standard_normal=None):
    if standard_normal is None:
        standard_normal = torch.tensor([0., 0., 1.]).reshape(1, 3).expand(plane_normal_init.shape[0], 3).to(plane_normal_init.device)
    angle_diff = torch.acos((standard_normal * plane_normal_init).sum(dim=-1).clamp(-1, 1)).reshape(-1, 1)
    rot_axis = torch.cross(standard_normal, plane_normal_init, dim=-1)  # n_plane, 3
    rot_axis = F.normalize(rot_axis, dim=-1)
    rot_vec = (rot_axis * angle_diff).cpu().numpy()
    rot_q = quaternion.as_float_array(quaternion.from_rotation_vector(rot_vec))
    rot_q = torch.from_numpy(rot_q).float()
    return rot_q

def quaternion_mult(q1, q2):
    '''
    q1 x q2

    q1 = w1+i*x1+j*y1+k*z1
    q2 = w2+i*x2+j*y2+k*z2
    q1*q2 =
     (w1w2 - x1x2 - y1y2 - z1z2)
    +(w1x2 + x1w2 + y1z2 - z1y2)i
    +(w1y2 - x1z2 + y1w2 + z1x2)j
    +(w1z2 + x1y2 - y1x2 + z1w2)k

    :param q1:
    :param q2:
    :return:
    '''

    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)

    w1 = q1[:, 0]  # bs
    x1 = q1[:, 1]
    y1 = q1[:, 2]
    z1 = q1[:, 3]

    w2 = q2[:, 0]  # bs
    x2 = q2[:, 1]
    y2 = q2[:, 2]
    z2 = q2[:, 3]

    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2

    q = torch.stack([w, x, y, z], dim=-1)

    return q