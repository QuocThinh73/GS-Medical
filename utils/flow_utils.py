import torch
import torch.nn.functional as F


def flow_warp(src: torch.Tensor, flow: torch.Tensor, mode="bilinear", padding_mode="zeros"):
    B, C, H, W = src.shape
    device = src.device

    y, x = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    )
    grid = torch.stack([x, y], dim=0).float()[None].repeat(B, 1, 1, 1)  # [B,2,H,W]
    coords = grid + flow

    x = coords[:, 0]
    y = coords[:, 1]
    x = 2.0 * (x / (W - 1.0)) - 1.0
    y = 2.0 * (y / (H - 1.0)) - 1.0
    grid_norm = torch.stack([x, y], dim=-1)  # [B,H,W,2]

    return F.grid_sample(src, grid_norm, mode=mode, padding_mode=padding_mode, align_corners=True)