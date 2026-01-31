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
from utils.flow_utils import flow_warp


def TV_loss(x, mask=None):
    B, C, H, W = x.shape
    tv_h = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).sum()
    tv_w = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).sum()
    return (tv_h + tv_w) / (B * C * H * W)


def lpips_loss(img1, img2, lpips_model):
    loss = lpips_model(img1,img2)
    return loss.mean()

def l1_loss(network_output, gt, mask=None):
    loss = torch.abs((network_output - gt))
    if mask is not None:
        if mask.ndim == 4:
            mask = mask.repeat(1, network_output.shape[1], 1, 1)
        elif mask.ndim == 3:
            mask = mask.repeat(network_output.shape[1], 1, 1)
        else:
            raise ValueError('the dimension of mask should be either 3 or 4')
    
        try:
            loss = loss[mask!=0]
        except:
            print(loss.shape)
            print(mask.shape)
            print(loss.dtype)
            print(mask.dtype)
    return loss.mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
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

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def unit_expos_loss(tone_mapper, gt):
    ln_x = torch.zeros([1,3]).cuda()
    rgb_l = tone_mapper(ln_x)
    
    return torch.mean((rgb_l - gt) ** 2)

def _in_bounds_mask(flow):
    B, _, H, W = flow.shape
    device = flow.device
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    grid = torch.stack([x, y], dim=0).float()[None].repeat(B, 1, 1, 1)
    coords = grid + flow
    x = coords[:, 0]
    y = coords[:, 1]
    ok = (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1))
    return ok[:, None]

def temporal_luminance_loss(img_src, img_dest, flow, mask_src, mask_dest):
    warp = flow_warp(img_src, flow, mode="bilinear", padding_mode="zeros")

    tissue_dest_at = flow_warp(mask_dest.float(), flow, mode="nearest", padding_mode="zeros") > 0.5

    inb = _in_bounds_mask(flow)

    valid_mask = (mask_src > 0.5) & tissue_dest_at & inb

    return l1_loss(warp, img_dest, valid_mask)