import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from scene.flexible_deform_model import ToneMapper


def load_state_dict_safely(model: nn.Module, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    return missing, unexpected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, required=True,)
    parser.add_argument("--ckpt_name", type=str, default="tone_mapper.pth")
    parser.add_argument("--out_name", type=str, default="tone_mapper_curve.png",)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--xmin", type=float, default=-5.0, help="Min exposure input")
    parser.add_argument("--xmax", type=float, default=5.0, help="Max exposure input")
    parser.add_argument("--num", type=int, default=2000, help="Number of sample points")
    parser.add_argument("--logx", action="store_true", help="Use log-spaced x (good if exposure spans decades)")
    args = parser.parse_args()

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(folder):
        raise NotADirectoryError(f"Folder not found: {folder}")

    ckpt_path = os.path.join(folder, args.ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path = os.path.join(folder, args.out_name)

    device = args.device
    model = ToneMapper().to(device).eval()

    load_state_dict_safely(model, ckpt_path, device)

    # Build exposure samples
    if args.logx:
        xmin = max(args.xmin, 1e-6)
        if args.xmax <= 0:
            raise ValueError("--xmax must be > 0 when using --logx")
        x = np.logspace(np.log10(xmin), np.log10(args.xmax), args.num).astype(np.float32)
    else:
        x = np.linspace(args.xmin, args.xmax, args.num, dtype=np.float32)

    # Same exposure for all channels
    x_t = torch.from_numpy(x).to(device)
    exposure = torch.stack([x_t, x_t, x_t], dim=-1)  # [N,3]

    with torch.no_grad():
        y = model(exposure).detach().cpu().numpy()  # [N,3] in [0,1]

    # Plot
    plt.figure()
    plt.plot(x, y[:, 0], label="R")
    plt.plot(x, y[:, 1], label="G")
    plt.plot(x, y[:, 2], label="B")
    plt.xlabel("Exposure input (proxy for HDR scale)")
    plt.ylabel("ToneMapper output (LDR in [0,1])")
    plt.title("ToneMapper transfer curves")
    plt.grid(True)
    plt.legend()

    if args.logx:
        plt.xscale("log")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print("Loaded:", ckpt_path)
    print("Saved chart:", out_path)


if __name__ == "__main__":
    main()
