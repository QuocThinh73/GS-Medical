import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def read_rgb01(path: str) -> torch.Tensor:
    """[1,3,H,W] float in [0,1]."""
    img = np.array(Image.open(path).convert("RGB"), dtype=np.uint8)
    t = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
    return t


def read_tool_mask(path: str) -> torch.Tensor:
    """[1,1,H,W] float: 1=tool, 0=not tool (non-zero in png means tool)."""
    m = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    tool = (m > 0).astype(np.float32)
    return torch.from_numpy(tool)[None, None]


def erode_binary(mask01: torch.Tensor, k: int = 3, iters: int = 1) -> torch.Tensor:
    if k <= 1 or iters <= 0:
        return mask01
    inv = 1.0 - mask01
    pad = k // 2
    for _ in range(iters):
        inv = F.max_pool2d(inv, kernel_size=k, stride=1, padding=pad)
    return 1.0 - inv


def coords_grid(B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    grid = torch.stack([x, y], dim=0).float()  # [2,H,W]
    return grid[None].repeat(B, 1, 1, 1)      # [B,2,H,W]


def flow_warp(src: torch.Tensor, flow: torch.Tensor, mode="bilinear", padding_mode="zeros") -> torch.Tensor:
    """
    src:  [B,C,H,W]
    flow: [B,2,H,W] in pixel units (src->dst)
    """
    B, C, H, W = src.shape
    device = src.device
    grid = coords_grid(B, H, W, device)
    coords = grid + flow

    x = coords[:, 0]
    y = coords[:, 1]
    x_norm = 2.0 * (x / (W - 1.0)) - 1.0
    y_norm = 2.0 * (y / (H - 1.0)) - 1.0
    grid_norm = torch.stack([x_norm, y_norm], dim=-1)  # [B,H,W,2]

    return F.grid_sample(src, grid_norm, mode=mode, padding_mode=padding_mode, align_corners=True)


def in_bounds_mask(flow: torch.Tensor) -> torch.Tensor:
    B, _, H, W = flow.shape
    device = flow.device
    grid = coords_grid(B, H, W, device)
    coords = grid + flow
    x = coords[:, 0]
    y = coords[:, 1]
    ok = (x >= 0) & (x <= (W - 1)) & (y >= 0) & (y <= (H - 1))
    return ok[:, None]


def sample_mask_nearest(mask01: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    return flow_warp(mask01, flow, mode="nearest", padding_mode="zeros")


def fb_consistency_mask(flow_fwd: torch.Tensor, flow_bwd: torch.Tensor, tau: float = 2.0) -> torch.Tensor:
    bwd_at_xprime = flow_warp(flow_bwd, flow_fwd, mode="bilinear", padding_mode="zeros")
    cycle = flow_fwd + bwd_at_xprime
    err = torch.sqrt(torch.sum(cycle * cycle, dim=1, keepdim=True) + 1e-8)
    return err <= tau


def load_flow_npy(path: str, device: torch.device) -> torch.Tensor:
    """Load [H,W,2] -> [1,2,H,W] float32"""
    f = np.load(path)
    assert f.ndim == 3 and f.shape[2] == 2, f"Bad flow shape {f.shape} in {path}"
    t = torch.from_numpy(f).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return t


def save_bool_png(mask_bool: torch.Tensor, path: Path):
    m = (mask_bool[0, 0].detach().cpu().numpy().astype(np.uint8) * 255)
    Image.fromarray(m).save(path)


def save_rgb01(img01: torch.Tensor, path: Path):
    arr = (img01[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


def save_gray01(gray01: torch.Tensor, path: Path):
    arr = (gray01[0, 0].clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(path)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()

    # images
    p.add_argument("--img_tm1", default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC/cutting_tissues_twice/images/000010.png")
    p.add_argument("--img_t",   default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC/cutting_tissues_twice/images/000011.png")
    p.add_argument("--img_tp1", default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC/cutting_tissues_twice/images/000012.png")

    # flows (precomputed)
    p.add_argument("--flow_to_next_t",  default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC/cutting_tissues_twice/optical_flow_to_next/000011.npy")
    p.add_argument("--flow_to_prev_t",  default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC/cutting_tissues_twice/optical_flow_to_prev/000011.npy")
    p.add_argument("--flow_to_prev_tp1", default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC/cutting_tissues_twice/optical_flow_to_prev/000012.npy")  # (t+1)->t
    p.add_argument("--flow_to_next_tm1", default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC/cutting_tissues_twice/optical_flow_to_next/000010.npy")  # (t-1)->t

    # tool masks
    p.add_argument("--mask_tm1", default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC/cutting_tissues_twice/masks/frame-000010.mask.png")
    p.add_argument("--mask_t",   default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC/cutting_tissues_twice/masks/frame-000011.mask.png")
    p.add_argument("--mask_tp1", default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF-EC/cutting_tissues_twice/masks/frame-000012.mask.png")

    # params
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--erode", type=int, default=3)
    p.add_argument("--erode_iters", type=int, default=1)
    p.add_argument("--cpu", action="store_true")

    # outputs
    p.add_argument("--out_dir", default="output/demo-warping")
    args = p.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load images
    I_tm1 = read_rgb01(args.img_tm1).to(device)
    I_t   = read_rgb01(args.img_t).to(device)
    I_tp1 = read_rgb01(args.img_tp1).to(device)

    # load flows
    f_t2tp1 = load_flow_npy(args.flow_to_next_t, device)    # t -> t+1
    f_t2tm1 = load_flow_npy(args.flow_to_prev_t, device)    # t -> t-1
    f_tp12t = load_flow_npy(args.flow_to_prev_tp1, device)  # t+1 -> t
    f_tm12t = load_flow_npy(args.flow_to_next_tm1, device)  # t-1 -> t

    # load masks (tool=1)
    tool_tm1 = read_tool_mask(args.mask_tm1).to(device)
    tool_t   = read_tool_mask(args.mask_t).to(device)
    tool_tp1 = read_tool_mask(args.mask_tp1).to(device)

    # tissue masks
    tissue_tm1 = 1.0 - tool_tm1
    tissue_t   = 1.0 - tool_t
    tissue_tp1 = 1.0 - tool_tp1

    if args.erode >= 3 and args.erode_iters > 0:
        tissue_tm1 = erode_binary(tissue_tm1, k=args.erode, iters=args.erode_iters)
        tissue_t   = erode_binary(tissue_t,   k=args.erode, iters=args.erode_iters)
        tissue_tp1 = erode_binary(tissue_tp1, k=args.erode, iters=args.erode_iters)

    # ---- valid mask t -> t+1 ----
    valid_next = (
        (tissue_t > 0.5)
        & in_bounds_mask(f_t2tp1)
        & (sample_mask_nearest(tissue_tp1, f_t2tp1) > 0.5)
        & fb_consistency_mask(f_t2tp1, f_tp12t, tau=args.tau)
    )

    # ---- valid mask t -> t-1 ----
    valid_prev = (
        (tissue_t > 0.5)
        & in_bounds_mask(f_t2tm1)
        & (sample_mask_nearest(tissue_tm1, f_t2tm1) > 0.5)
        & fb_consistency_mask(f_t2tm1, f_tm12t, tau=args.tau)
    )

    # ---- warp + error ----
    warp_to_next = flow_warp(I_t, f_t2tp1, mode="bilinear")
    warp_to_prev = flow_warp(I_t, f_t2tm1, mode="bilinear")

    # photometric error (L1 on RGB, then mean over channels)
    err_next = (warp_to_next - I_tp1).abs().mean(dim=1, keepdim=True)  # [1,1,H,W]
    err_prev = (warp_to_prev - I_tm1).abs().mean(dim=1, keepdim=True)

    # mask invalid pixels out (set to 0 in visualization; and compute masked mean)
    err_next_masked = err_next * valid_next.float()
    err_prev_masked = err_prev * valid_prev.float()

    # normalize error for display (robust)
    def norm01(x, eps=1e-6):
        # scale by 99th percentile to avoid a few outliers dominating
        v = x[x > 0]
        if v.numel() == 0:
            return x
        q = torch.quantile(v, 0.99)
        return torch.clamp(x / (q + eps), 0.0, 1.0)

    err_next_vis = norm01(err_next_masked)
    err_prev_vis = norm01(err_prev_masked)

    # save outputs
    save_bool_png(valid_next, out_dir / "valid_to_next.png")
    save_bool_png(valid_prev, out_dir / "valid_to_prev.png")

    save_rgb01(warp_to_next, out_dir / "warp_t_to_next.png")
    save_rgb01(warp_to_prev, out_dir / "warp_t_to_prev.png")

    save_gray01(err_next_vis, out_dir / "error_to_next.png")
    save_gray01(err_prev_vis, out_dir / "error_to_prev.png")

    # print masked mean error (only valid pixels)
    def masked_mean(err, valid):
        v = valid.float()
        denom = v.sum().clamp_min(1.0)
        return (err * v).sum() / denom

    mean_err_next = masked_mean(err_next, valid_next).item()
    mean_err_prev = masked_mean(err_prev, valid_prev).item()

    H, W = valid_next.shape[-2:]
    print(f"[OK] saved to {out_dir}")
    print(f" valid_to_next: {int(valid_next.sum())}/{H*W} | mean_L1_next={mean_err_next:.6f}")
    print(f" valid_to_prev: {int(valid_prev.sum())}/{H*W} | mean_L1_prev={mean_err_prev:.6f}")


if __name__ == "__main__":
    main()
