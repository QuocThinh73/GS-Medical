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

def readImages(renders_dir, gt_dir, renders_inpaint_dir, gt_inpaint_dir, depth_dir, gtdepth_dir, masks_dir):
    renders = []
    gts = []

    renders_inpaint = []
    gts_inpaint = []

    image_names = []
    depths = []
    gt_depths = []
    masks = []
    
    for fname in os.listdir(renders_dir):
        render = np.array(Image.open(renders_dir / fname))
        gt = np.array(Image.open(gt_dir / fname))

        render_inpaint = np.array(Image.open(renders_inpaint_dir / fname))
        gt_inpaint = np.array(Image.open(gt_inpaint_dir / fname))

        depth = np.array(Image.open(depth_dir / fname))
        gt_depth = np.array(Image.open(gtdepth_dir / fname))
        mask = np.array(Image.open(masks_dir / fname))
        
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())

        renders_inpaint.append(tf.to_tensor(render_inpaint).unsqueeze(0)[:, :3, :, :].cuda())
        gts_inpaint.append(tf.to_tensor(gt_inpaint).unsqueeze(0)[:, :3, :, :].cuda())

        depths.append(torch.from_numpy(depth).unsqueeze(0).unsqueeze(1).cuda())
        gt_depths.append(torch.from_numpy(gt_depth).unsqueeze(0).unsqueeze(1).cuda())
        masks.append(tf.to_tensor(mask).unsqueeze(0).cuda())
        
        image_names.append(fname)

    return renders, gts, renders_inpaint, gts_inpaint, depths, gt_depths, masks, image_names

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

            test_dir = Path(scene_dir) / args.phase

            for method in os.listdir(test_dir):
                print("Method:", method)

                full_dict[scene_dir][method] = {}
                per_view_dict[scene_dir][method] = {}
                full_dict_polytopeonly[scene_dir][method] = {}
                per_view_dict_polytopeonly[scene_dir][method] = {}

                method_dir = test_dir / method
                gt_dir = method_dir / "gt"
                gt_inpaint_dir = method_dir / "gt_inpaint"
                renders_dir = method_dir / "renders"
                renders_inpaint_dir = method_dir / "renders_inpaint"
                depth_dir = method_dir / "depth"
                gt_depth_dir = method_dir / "gt_depth"
                masks_dir = method_dir / "masks"
                
                renders, gts, renders_inpaint, gts_inpaint, depths, gt_depths, masks, image_names = readImages(
                    renders_dir, gt_dir,
                    renders_inpaint_dir, gt_inpaint_dir,
                    depth_dir, gt_depth_dir, masks_dir
                )

                final_ssims = []
                final_psnrs = []
                final_lpipss = []

                inpaint_ssims = []
                inpaint_psnrs = []
                inpaint_lpipss = []

                rmses = []
                                
                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    render = renders[idx]
                    gt = gts[idx]

                    render_inpaint = renders_inpaint[idx]
                    gt_inpaint = gts_inpaint[idx]

                    depth = depths[idx]
                    gt_depth = gt_depths[idx]
                    mask = masks[idx]

                    # final branch
                    render_masked = render * mask
                    gt_masked = gt * mask

                    final_psnrs.append(psnr(render_masked, gt_masked))
                    final_ssims.append(ssim(render_masked, gt_masked))
                    final_lpipss.append(cal_lpips(render_masked, gt_masked))

                    # inpaint branch
                    render_inpaint_masked = render_inpaint * mask
                    gt_inpaint_masked = gt_inpaint * mask

                    inpaint_psnrs.append(psnr(render_inpaint_masked, gt_inpaint_masked))
                    inpaint_ssims.append(ssim(render_inpaint_masked, gt_inpaint_masked))
                    inpaint_lpipss.append(cal_lpips(render_inpaint_masked, gt_inpaint_masked))

                    # depth
                    if (gt_depth != 0).sum() >= 10:
                        rmses.append(rmse(depth, gt_depth, mask))

                print("Scene: ", scene_dir, "FINAL SSIM : {:>12.7f}".format(torch.tensor(final_ssims).mean(), ".5"))
                print("Scene: ", scene_dir, "FINAL PSNR : {:>12.7f}".format(torch.tensor(final_psnrs).mean(), ".5"))
                print("Scene: ", scene_dir, "FINAL LPIPS: {:>12.7f}".format(torch.tensor(final_lpipss).mean(), ".5"))

                print("Scene: ", scene_dir, "INPAINT SSIM : {:>12.7f}".format(torch.tensor(inpaint_ssims).mean(), ".5"))
                print("Scene: ", scene_dir, "INPAINT PSNR : {:>12.7f}".format(torch.tensor(inpaint_psnrs).mean(), ".5"))
                print("Scene: ", scene_dir, "INPAINT LPIPS: {:>12.7f}".format(torch.tensor(inpaint_lpipss).mean(), ".5"))

                print("Scene: ", scene_dir, "RMSE: {:>12.7f}".format(torch.tensor(rmses).mean(), ".5"))
                print("")

                full_dict[scene_dir][method].update({
                    "FINAL_SSIM": torch.tensor(final_ssims).mean().item(),
                    "FINAL_PSNR": torch.tensor(final_psnrs).mean().item(),
                    "FINAL_LPIPS": torch.tensor(final_lpipss).mean().item(),

                    "INPAINT_SSIM": torch.tensor(inpaint_ssims).mean().item(),
                    "INPAINT_PSNR": torch.tensor(inpaint_psnrs).mean().item(),
                    "INPAINT_LPIPS": torch.tensor(inpaint_lpipss).mean().item(),

                    "RMSE": torch.tensor(rmses).mean().item() if len(rmses) > 0 else None
                })
                per_view_dict[scene_dir][method].update({
                    "FINAL_SSIM": {name: val for val, name in zip(torch.tensor(final_ssims).tolist(), image_names)},
                    "FINAL_PSNR": {name: val for val, name in zip(torch.tensor(final_psnrs).tolist(), image_names)},
                    "FINAL_LPIPS": {name: val for val, name in zip(torch.tensor(final_lpipss).tolist(), image_names)},

                    "INPAINT_SSIM": {name: val for val, name in zip(torch.tensor(inpaint_ssims).tolist(), image_names)},
                    "INPAINT_PSNR": {name: val for val, name in zip(torch.tensor(inpaint_psnrs).tolist(), image_names)},
                    "INPAINT_LPIPS": {name: val for val, name in zip(torch.tensor(inpaint_lpipss).tolist(), image_names)},

                    "RMSE": {name: val for val, name in zip(torch.tensor(rmses).tolist(), image_names[:len(rmses)])}
                })

            with open(scene_dir + "/results.json", 'w') as fp:
                json.dump(full_dict[scene_dir], fp, indent=True)
            with open(scene_dir + "/per_view.json", 'w') as fp:
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
