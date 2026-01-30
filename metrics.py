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

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
# from lpipsPyTorch import lpips
import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from utils.image_utils import rmse
from argparse import ArgumentParser
import numpy as np


def array2tensor(array, device="cuda", dtype=torch.float32):
    return torch.tensor(array, dtype=dtype, device=device)

# Learned Perceptual Image Patch Similarity
class LPIPS(object):
    """
    borrowed from https://github.com/huster-wgm/Pytorch-metrics/blob/master/metrics.py
    """
    def __init__(self, device="cuda"):
        self.model = lpips.LPIPS(net='alex').to(device)

    def __call__(self, y_pred, y_true, normalized=True):
        """
        args:
            y_true : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            y_pred : 4-d ndarray in [batch_size, channels, img_rows, img_cols]
            normalized : change [0,1] => [-1,1] (default by LPIPS)
        return LPIPS, smaller the better
        """
        if normalized:
            y_pred = y_pred * 2.0 - 1.0
            y_true = y_true * 2.0 - 1.0
        error =  self.model.forward(y_pred, y_true)
        return torch.mean(error)
    
lpips = LPIPS()
def cal_lpips(a, b, device="cuda", batch=2):
    """Compute lpips.
    a, b: [batch, H, W, 3]"""
    if not torch.is_tensor(a):
        a = array2tensor(a, device)
    if not torch.is_tensor(b):
        b = array2tensor(b, device)

    lpips_all = []
    for a_split, b_split in zip(a.split(split_size=batch, dim=0), b.split(split_size=batch, dim=0)):
        out = lpips(a_split, b_split)
        lpips_all.append(out)
    lpips_all = torch.stack(lpips_all)
    lpips_mean = lpips_all.mean()
    return lpips_mean

def readImages(consistent_ldr_from2d_renders_dir, consistent_ldr_from3d_renders_dir, normal_gts_dir, ldr_from2d_renders_dir, ldr_from3d_renders_dir, gts_dir, depth_dir, depth_gts_dir, masks_dir):
    consistent_ldr_from2d_renders = []
    consistent_ldr_from3d_renders = []
    normal_gts = []
    ldr_from2d_renders = []
    ldr_from3d_renders = []
    gts = []
    depths = []
    depth_gts = []
    masks = []
    image_names = []
    
    for fname in os.listdir(gts_dir):
        consistent_ldr_from2d_render = np.array(Image.open(consistent_ldr_from2d_renders_dir / fname))
        consistent_ldr_from3d_render = np.array(Image.open(consistent_ldr_from3d_renders_dir / fname))
        normal_gt = np.array(Image.open(normal_gts_dir / fname))
        ldr_from2d_render = np.array(Image.open(ldr_from2d_renders_dir / fname))
        ldr_from3d_render = np.array(Image.open(ldr_from3d_renders_dir / fname))
        gt = np.array(Image.open(gts_dir / fname))
        depth = np.array(Image.open(depth_dir / fname))
        depth_gt = np.array(Image.open(depth_gts_dir / fname))
        mask = np.array(Image.open(masks_dir / fname))
        
        consistent_ldr_from2d_renders.append(tf.to_tensor(consistent_ldr_from2d_render).unsqueeze(0)[:, :3, :, :].cuda())
        consistent_ldr_from3d_renders.append(tf.to_tensor(consistent_ldr_from3d_render).unsqueeze(0)[:, :3, :, :].cuda())
        normal_gts.append(tf.to_tensor(normal_gt).unsqueeze(0)[:, :3, :, :].cuda())
        ldr_from2d_renders.append(tf.to_tensor(ldr_from2d_render).unsqueeze(0)[:, :3, :, :].cuda())
        ldr_from3d_renders.append(tf.to_tensor(ldr_from3d_render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        depths.append(torch.from_numpy(depth).unsqueeze(0).unsqueeze(1)[:, :, :, :].cuda())
        depth_gts.append(torch.from_numpy(depth_gt).unsqueeze(0).unsqueeze(1)[:, :3, :, :].cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0).cuda())
        
        image_names.append(fname)

    return consistent_ldr_from2d_renders, consistent_ldr_from3d_renders, normal_gts, ldr_from2d_renders, ldr_from3d_renders, gts, depths, depth_gts, masks, image_names

def evaluate(model_paths):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    with torch.no_grad():
        for scene_dir in model_paths:
            print("Scene:", scene_dir)
            full_dict[scene_dir] = {}
            per_view_dict[scene_dir] = {}
            full_dict_polytopeonly[scene_dir] = {}
            per_view_dict_polytopeonly[scene_dir] = {}

            evaluation_dir = Path(scene_dir) / args.phase

            for method in os.listdir(evaluation_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = evaluation_dir / method
                consistent_ldr_from2d_renders_dir = method_dir / "consistent_ldr_from2d_renders"
                consistent_ldr_from3d_renders_dir = method_dir / "consistent_ldr_from3d_renders"
                normal_gts_dir = method_dir/ "normal_gt"
                ldr_from2d_renders_dir = method_dir / "ldr_from2d_renders"
                ldr_from3d_renders_dir = method_dir / "ldr_from3d_renders"
                gts_dir = method_dir/ "gt"
                depth_dir = method_dir / "depth"
                depth_gts_dir = method_dir / "gt_depth"
                masks_dir = method_dir / "masks"
                
                consistent_ldr_from2d_renders, consistent_ldr_from3d_renders, normal_gts, ldr_from2d_renders, ldr_from3d_renders, gts, depths, depth_gts, masks, image_names = \
                    readImages(consistent_ldr_from2d_renders_dir, consistent_ldr_from3d_renders_dir, normal_gts_dir, ldr_from2d_renders_dir, ldr_from3d_renders_dir, gts_dir, depth_dir, depth_gts_dir, masks_dir)

                consistent_ldr_from2d_ssims = []
                consistent_ldr_from2d_psnrs = []
                consistent_ldr_from2d_lpipss = []
                consistent_ldr_from3d_ssims = []
                consistent_ldr_from3d_psnrs = []
                consistent_ldr_from3d_lpipss = []
                if args.phase == "train":
                    ldr_from2d_ssims = []
                    ldr_from2d_psnrs = []
                    ldr_from2d_lpipss = []
                    ldr_from3d_ssims = []
                    ldr_from3d_psnrs = []
                    ldr_from3d_lpipss = []
                rmses = []
                                
                for idx in tqdm(range(len(gts)), desc="Metric evaluation progress"):
                    consistent_ldr_from2d_render, consistent_ldr_from3d_render, normal_gt, ldr_from2d_render, ldr_from3d_render, gt, depth, depth_gt, mask, image_name = (
                        consistent_ldr_from2d_renders[idx], 
                        consistent_ldr_from3d_renders[idx], 
                        normal_gts[idx], 
                        ldr_from2d_renders[idx], 
                        ldr_from3d_renders[idx], 
                        gts[idx], 
                        depths[idx], 
                        depth_gts[idx], 
                        masks[idx], 
                        image_names[idx]
                    )
                    consistent_ldr_from2d_render = consistent_ldr_from2d_render * mask
                    consistent_ldr_from3d_render = consistent_ldr_from3d_render * mask
                    normal_gt = normal_gt * mask
                    
                    consistent_ldr_from2d_psnrs.append(psnr(consistent_ldr_from2d_render, normal_gt))
                    consistent_ldr_from2d_ssims.append(ssim(consistent_ldr_from2d_render, normal_gt))
                    consistent_ldr_from2d_lpipss.append(cal_lpips(consistent_ldr_from2d_render, normal_gt))
                    consistent_ldr_from3d_psnrs.append(psnr(consistent_ldr_from3d_render, normal_gt))
                    consistent_ldr_from3d_ssims.append(ssim(consistent_ldr_from3d_render, normal_gt))
                    consistent_ldr_from3d_lpipss.append(cal_lpips(consistent_ldr_from3d_render, normal_gt))

                    if args.phase == "train":
                        ldr_from2d_render = ldr_from2d_render * mask
                        ldr_from3d_render = ldr_from3d_render * mask
                        gt = gt * mask

                        ldr_from2d_ssims.append(psnr(ldr_from2d_render, gt))
                        ldr_from2d_psnrs.append(ssim(ldr_from2d_render, gt))
                        ldr_from2d_lpipss.append(cal_lpips(ldr_from2d_render, gt))
                        ldr_from3d_ssims.append(psnr(ldr_from3d_render, gt))
                        ldr_from3d_psnrs.append(ssim(ldr_from3d_render, gt))
                        ldr_from3d_lpipss.append(cal_lpips(ldr_from3d_render, gt))

                    if (depth_gt!=0).sum() < 10:
                        continue
                    rmses.append(rmse(depth, depth_gt, mask))

                print("Scene: ", scene_dir,  "Consistent LDR from 2D image SSIM : {:>12.7f}".format(torch.tensor(consistent_ldr_from2d_ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "Consistent LDR from 2D image PSNR : {:>12.7f}".format(torch.tensor(consistent_ldr_from2d_psnrs).mean(), ".5"))
                print("Scene: ", scene_dir,  "Consistent LDR from 2D image LPIPS: {:>12.7f}".format(torch.tensor(consistent_ldr_from2d_lpipss).mean(), ".5"))
                print("Scene: ", scene_dir,  "Consistent LDR from 3D image SSIM : {:>12.7f}".format(torch.tensor(consistent_ldr_from3d_ssims).mean(), ".5"))
                print("Scene: ", scene_dir,  "Consistent LDR from 3D image PSNR : {:>12.7f}".format(torch.tensor(consistent_ldr_from3d_psnrs).mean(), ".5"))
                print("Scene: ", scene_dir,  "Consistent LDR from 3D image LPIPS: {:>12.7f}".format(torch.tensor(consistent_ldr_from3d_lpipss).mean(), ".5"))
                if args.phase == "train":
                    print("Scene: ", scene_dir,  "LDR from 2D image SSIM : {:>12.7f}".format(torch.tensor(ldr_from2d_ssims).mean(), ".5"))
                    print("Scene: ", scene_dir,  "LDR from 2D image PSNR : {:>12.7f}".format(torch.tensor(ldr_from2d_psnrs).mean(), ".5"))
                    print("Scene: ", scene_dir,  "LDR from 2D image LPIPS: {:>12.7f}".format(torch.tensor(ldr_from2d_lpipss).mean(), ".5"))
                    print("Scene: ", scene_dir,  "LDR from 3D image SSIM : {:>12.7f}".format(torch.tensor(ldr_from3d_ssims).mean(), ".5"))
                    print("Scene: ", scene_dir,  "LDR from 3D image PSNR : {:>12.7f}".format(torch.tensor(ldr_from3d_psnrs).mean(), ".5"))
                    print("Scene: ", scene_dir,  "LDR from 3D image LPIPS: {:>12.7f}".format(torch.tensor(ldr_from3d_lpipss).mean(), ".5"))
                print("Scene: ", scene_dir,  "Depth map RMSE: {:>12.7f}".format(torch.tensor(rmses).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({"Consistent LDR from 2D image SSIM": torch.tensor(consistent_ldr_from2d_ssims).mean().item(),
                                                        "Consistent LDR from 2D image PSNR": torch.tensor(consistent_ldr_from2d_psnrs).mean().item(),
                                                        "Consistent LDR from 2D image LPIPS": torch.tensor(consistent_ldr_from2d_lpipss).mean().item(),
                                                        "Consistent LDR from 3D image SSIM": torch.tensor(consistent_ldr_from3d_ssims).mean().item(),
                                                        "Consistent LDR from 3D image PSNR": torch.tensor(consistent_ldr_from3d_psnrs).mean().item(),
                                                        "Consistent LDR from 3D image LPIPS": torch.tensor(consistent_ldr_from3d_lpipss).mean().item()})
                per_view_dict[scene_dir][method].update({"Consistent LDR from 2D image SSIMS": {name: ssim for ssim, name in zip(torch.tensor(consistent_ldr_from2d_ssims).tolist(), image_names)},
                                                            "Consistent LDR from 2D image PSNRS": {name: psnr for psnr, name in zip(torch.tensor(consistent_ldr_from2d_psnrs).tolist(), image_names)},
                                                            "Consistent LDR from 2D image LPIPSS": {name: lp for lp, name in zip(torch.tensor(consistent_ldr_from2d_lpipss).tolist(), image_names)},
                                                            "Consistent LDR from 3D image SSIMS": {name: ssim for ssim, name in zip(torch.tensor(consistent_ldr_from3d_ssims).tolist(), image_names)},
                                                            "Consistent LDR from 3D image PSNRS": {name: psnr for psnr, name in zip(torch.tensor(consistent_ldr_from3d_psnrs).tolist(), image_names)},
                                                            "Consistent LDR from 3D image LPIPSS": {name: lp for lp, name in zip(torch.tensor(consistent_ldr_from3d_lpipss).tolist(), image_names)}})
                
                if args.phase == "train":
                    full_dict[scene_dir][method].update({"LDR from 2D image SSIM": torch.tensor(ldr_from2d_ssims).mean().item(),
                                                        "LDR from 2D image PSNR": torch.tensor(ldr_from2d_psnrs).mean().item(),
                                                        "LDR from 2D image LPIPS": torch.tensor(ldr_from2d_lpipss).mean().item(),
                                                        "LDR from 3D image SSIM": torch.tensor(ldr_from3d_ssims).mean().item(),
                                                        "LDR from 3D image PSNR": torch.tensor(ldr_from3d_psnrs).mean().item(),
                                                        "LDR from 3D image LPIPS": torch.tensor(ldr_from3d_lpipss).mean().item()})
                    
                    per_view_dict[scene_dir][method].update({"LDR from 2D image SSIMS": {name: ssim for ssim, name in zip(torch.tensor(ldr_from2d_ssims).tolist(), image_names)},
                                                            "LDR from 2D image PSNRS": {name: psnr for psnr, name in zip(torch.tensor(ldr_from2d_psnrs).tolist(), image_names)},
                                                            "LDR from 2D image LPIPSS": {name: lp for lp, name in zip(torch.tensor(ldr_from2d_lpipss).tolist(), image_names)},
                                                            "LDR from 3D image SSIMS": {name: ssim for ssim, name in zip(torch.tensor(ldr_from3d_ssims).tolist(), image_names)},
                                                            "LDR from 3D image PSNRS": {name: psnr for psnr, name in zip(torch.tensor(ldr_from3d_psnrs).tolist(), image_names)},
                                                            "LDR from 3D image LPIPSS": {name: lp for lp, name in zip(torch.tensor(ldr_from3d_lpipss).tolist(), image_names)}})
                    
                full_dict[scene_dir][method].update({"Depth map RMSE": torch.tensor(rmses).mean().item()})
                per_view_dict[scene_dir][method].update({"Depth map RMSES": {name: lp for lp, name in zip(torch.tensor(rmses).tolist(), image_names)}})

            with open(scene_dir + f"/{args.phase}_overall_results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + f"/{args.phase}_per_view_results.json", 'w') as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--model_paths', '-m', required=True, nargs="+", type=str, default=[])
    parser.add_argument('--phase', '-p', type=str, default='test')
    args = parser.parse_args()
    evaluate(args.model_paths)
