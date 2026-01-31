import sys
import os
import glob
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.append("submodules/RAFT")

from core.raft import RAFT
from core.utils.utils import InputPadder
from core.utils import flow_viz


IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def list_images(folder: Path):
    files = []
    for ext in IMG_EXTS:
        files += glob.glob(str(folder / f"*{ext}"))
    return sorted(files)


def load_image(imfile: str, device: str):
    img = np.array(Image.open(imfile).convert("RGB")).astype(np.uint8)  # [H,W,3]
    img = torch.from_numpy(img).permute(2, 0, 1).float()               # [3,H,W]
    return img[None].to(device)                                        # [1,3,H,W]


def flow_tensor_to_hw2_numpy(flow: torch.Tensor, dtype=np.float32) -> np.ndarray:
    """
    flow: [1,2,H,W] torch -> [H,W,2] numpy
    """
    flo = flow[0].permute(1, 2, 0).detach().cpu().numpy()
    if dtype == np.float16:
        return flo.astype(np.float16)
    return flo.astype(np.float32)


def flow_hw2_to_vis_png(flow_hw2: np.ndarray) -> np.ndarray:
    """
    flow_hw2: [H,W,2] -> uint8 RGB [H,W,3]
    """
    return flow_viz.flow_to_image(flow_hw2.astype(np.float32))


@torch.no_grad()
def main(args):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"

    scene_dir = Path(args.scene_dir).resolve()
    images_dir = scene_dir / args.images_subdir
    assert images_dir.exists(), f"Missing images folder: {images_dir}"

    # Output dirs (4 outputs)
    out_next = scene_dir / "optical_flow_to_next"
    out_prev = scene_dir / "optical_flow_to_prev"
    out_vis_next = scene_dir / "optical_flow_vis_to_next"
    out_vis_prev = scene_dir / "optical_flow_vis_to_prev"
    for d in [out_next, out_prev, out_vis_next, out_vis_prev]:
        d.mkdir(parents=True, exist_ok=True)

    images = list_images(images_dir)
    assert len(images) >= 2, f"Need at least 2 images in {images_dir}"

    # Build RAFT args
    raft_args = {
        "small": args.small,
        "mixed_precision": args.mixed_precision,
        "alternate_corr": args.alternate_corr,
        "dropout": 0.0, 
    }

    model = torch.nn.DataParallel(RAFT(raft_args))
    ckpt = torch.load(args.model, map_location=device)
    # Some checkpoints may store dict with keys; handle both
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=True)

    model = model.module
    model.to(device)
    model.eval()

    save_dtype = np.float16 if args.save_fp16 else np.float32

    print(f"[RAFT] device={device}")
    print(f"[RAFT] images={len(images)} from {images_dir}")
    print(f"[RAFT] iters={args.iters} | save_dtype={save_dtype.__name__}")
    print(f"[OUT] {out_next}\n[OUT] {out_prev}\n[OUT] {out_vis_next}\n[OUT] {out_vis_prev}")

    # For each consecutive pair (t, t+1)
    # - save forward to_next for t
    # - save backward to_prev for t+1
    for t in range(len(images) - 1):
        im_t = images[t]
        im_tp1 = images[t + 1]

        image_t = load_image(im_t, device)
        image_tp1 = load_image(im_tp1, device)

        padder = InputPadder(image_t.shape)
        image_t, image_tp1 = padder.pad(image_t, image_tp1)

        # Forward: t -> t+1
        flow_low_f, flow_up_f = model(image_t, image_tp1, iters=args.iters, test_mode=True)
        flow_hw2_f = flow_tensor_to_hw2_numpy(flow_up_f, dtype=save_dtype)
        np.save(out_next / f"{t:06d}.npy", flow_hw2_f)

        if args.save_vis:
            vis_f = flow_hw2_to_vis_png(flow_hw2_f)
            Image.fromarray(vis_f).save(out_vis_next / f"{t:06d}.png")

        # Backward: t+1 -> t  (this is "to_prev" for frame t+1)
        flow_low_b, flow_up_b = model(image_tp1, image_t, iters=args.iters, test_mode=True)
        flow_hw2_b = flow_tensor_to_hw2_numpy(flow_up_b, dtype=save_dtype)
        np.save(out_prev / f"{t+1:06d}.npy", flow_hw2_b)

        if args.save_vis:
            vis_b = flow_hw2_to_vis_png(flow_hw2_b)
            Image.fromarray(vis_b).save(out_vis_prev / f"{t+1:06d}.png")

        if (t + 1) % args.log_every == 0 or t == 0:
            print(f"  [{t+1}/{len(images)-1}] saved: next[{t:06d}] and prev[{t+1:06d}]")

    print("Done.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir", required=True, help="Folder containing images subfolder")
    p.add_argument("--images_subdir", default="images", help="Subfolder name inside scene_dir (default: images)")
    p.add_argument("--model", default="submodules/RAFT/pretrained/raft-things.pth", help="RAFT checkpoint path, e.g. models/raft-sintel.pth")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--small", action="store_true")
    p.add_argument("--mixed_precision", action="store_true")
    p.add_argument("--alternate_corr", action="store_true")
    p.add_argument("--cpu", action="store_true", help="Force CPU (very slow)")
    p.add_argument("--save_fp16", action="store_true", help="Save flow as float16 to reduce disk")
    p.add_argument("--save_vis", action="store_true", help="Also save visualization PNGs")
    p.add_argument("--log_every", type=int, default=25)
    args = p.parse_args()

    main(args)
