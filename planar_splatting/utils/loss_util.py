import torch
import torch.nn.functional as F

def metric_depth_loss(depth_pred, depth_gt, mask, max_depth=4.0, weight=None):
    depth_mask = torch.logical_and(depth_gt<=max_depth, depth_gt>0)
    depth_mask = torch.logical_and(depth_mask, mask)
    if depth_mask.sum() == 0:
        depth_loss = torch.tensor([0.]).mean().cuda()
    else:
        if weight is None:
            depth_loss = torch.mean(torch.abs((depth_pred - depth_gt)[depth_mask]))
        else:
            depth_loss = torch.mean((weight * torch.abs(depth_pred - depth_gt))[depth_mask])
    return depth_loss

def normal_loss_weights(normal_pred, normal_gt, mask, boundry_mask, weight = 2.0):
    weights = torch.ones(boundry_mask.shape, device=normal_gt.device) * 0.5
    weights[boundry_mask] = weight
    weights_flat = weights.reshape(-1, 1).expand(-1, 3)
    normal_pred = F.normalize(normal_pred, dim=-1)
    normal_gt = F.normalize(normal_gt, dim=-1)
    l1 = ((weights_flat * torch.abs(normal_pred - normal_gt)).sum(dim=-1))[mask].mean()
    cos = (weights.reshape(-1) * (1. - torch.sum(normal_pred * normal_gt, dim=-1)))[mask].mean()
    return l1, cos

def normal_loss_weights_patch(normal_pred, normal_gt, mask, boundry_mask, weight = 0.0):
    weights = torch.ones(boundry_mask.shape, device=normal_gt.device) 
    weights[boundry_mask] = weight
    weights_flat = weights.reshape(-1, 1).expand(-1, 3)
    normal_pred = F.normalize(normal_pred, dim=-1)
    normal_gt = F.normalize(normal_gt, dim=-1)
    l1 = ((weights_flat * torch.abs(normal_pred - normal_gt)).sum(dim=-1))[mask].mean()
    cos = (weights.reshape(-1) * (1. - torch.sum(normal_pred * normal_gt, dim=-1)))[mask].mean()
    return l1, cos

def normal_loss(normal_pred, normal_gt, mask):
    normal_pred = F.normalize(normal_pred, dim=-1)
    normal_gt = F.normalize(normal_gt, dim=-1)
    l1 = torch.abs(normal_pred - normal_gt).sum(dim=-1)[mask].mean()
    cos = (1. - torch.sum(normal_pred * normal_gt, dim=-1))[mask].mean()
    return l1, cos