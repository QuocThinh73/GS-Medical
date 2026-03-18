#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from pytorch3d.ops.knn import knn_points
from utils.image_utils import erode


def l1_loss(network_output, gt, mask=None):
    loss = torch.abs((network_output - gt))
    if mask is not None:
        mask = mask.repeat(network_output.shape[0], 1, 1)
        loss = loss[mask!=0]

    return loss.mean()

def l2_loss(network_output, gt, mask=None):
    loss = ((network_output - gt) ** 2)

    if mask is not None:
        mask = mask.repeat(network_output.shape[0], 1, 1)
        loss = loss[mask!=0]

    return loss.mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True, mask=None):
    img2 = img2.to(img1.dtype)

    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average, mask)

def _ssim(img1, img2, window, window_size, channel, size_average=True, mask=None, eps=1e-8):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if mask is not None:
        mask = mask.to(ssim_map.dtype)
        mask = mask.expand_as(ssim_map)

        return (ssim_map * mask).sum() / mask.sum().clamp_min(eps)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def TV_loss(x):
    C, H, W = x.shape
    tv_h = torch.abs(x[:, 1:, :] - x[:, :-1, :]).sum()
    tv_w = torch.abs(x[:, :, 1:] - x[:, :, :-1]).sum()
    return (tv_h + tv_w) / (C * H * W)

def def_reg_loss(gs_can, d_xyz, d_rotation, d_scaling, K=5):
    xyz_can = gs_can.get_xyz
    xyz_obs = xyz_can + d_xyz

    cov_can = gs_can.get_covariance()
    cov_obs = gs_can.get_covariance_obs(d_rotation, d_scaling)

    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0), xyz_can.unsqueeze(0), K=K, return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    dis_xyz_can = torch.cdist(xyz_can.unsqueeze(1), xyz_can[nn_ix])[:, 0, 1:]
    dis_xyz_obs = torch.cdist(xyz_obs.unsqueeze(1), xyz_obs[nn_ix])[:, 0, 1:]
    loss_pos = F.l1_loss(dis_xyz_can, dis_xyz_obs)

    dis_cov_can = torch.cdist(cov_can.unsqueeze(1), cov_can[nn_ix])[:, 0, 1:]
    dis_cov_obs = torch.cdist(cov_obs.unsqueeze(1), cov_obs[nn_ix])[:, 0, 1:]
    loss_cov = F.l1_loss(dis_cov_can, dis_cov_obs)

    return loss_pos, loss_cov

def compute_normal_loss(pred_normal, ori_normal, mask=None):
    # pred_normal: (3, H, W), ori_normal: (3, H, W)

    pred_normal = pred_normal.permute(1, 2, 0).reshape(-1, 3)
    ori_normal = ori_normal.permute(1, 2, 0).reshape(-1, 3).detach()
    loss = (1.0 - torch.sum(pred_normal * ori_normal, axis=-1))

    if mask is not None:
        mask = mask.reshape(-1).to(loss.device)
        loss = loss[mask!=0]

    return loss.mean()

def compute_depth_loss(pred_depth, gt_depth, mask=None, eps=1e-6):
    """
        pred_depth: (H, W)
        gt_depth:   (H, W)
        mask:       (H, W), 1 = hợp lệ, 0 = bỏ qua
    """
    pred_depth = pred_depth.float()
    gt_depth = gt_depth.float()

    valid = torch.isfinite(pred_depth) & torch.isfinite(gt_depth)
    valid = valid & (pred_depth > 0) & (gt_depth > 0)

    if mask is not None:
        valid = valid & (mask > 0)

    if valid.sum() == 0:
        return torch.tensor(0.0, device=pred_depth.device, dtype=pred_depth.dtype)
    
    pred_valid = pred_depth[valid]
    gt_valid = gt_depth[valid]

    t_pred = torch.median(pred_valid)
    s_pred = torch.mean(torch.abs(pred_valid - t_pred)).clamp_min(eps)
    pred_norm = (pred_valid - t_pred) / s_pred

    t_gt = torch.median(gt_valid)
    s_gt = torch.mean(torch.abs(gt_valid - t_gt)).clamp_min(eps)
    gt_norm = (gt_valid - t_gt) / s_gt

    return torch.mean((pred_norm - gt_norm) ** 2)