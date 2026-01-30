import os
import glob
import numpy as np
from PIL import Image, ImageDraw
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


# ------------------------
# IO
# ------------------------

def list_images(folder):
    files = []
    for ext in IMG_EXTS:
        files.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    return sorted(files)


def read_rgb_float(path):
    return np.asarray(Image.open(path).convert("RGB"), np.float32) / 255.0


def read_mask_bool(path, target_hw):
    m = Image.open(path).convert("L")
    if m.size != (target_hw[1], target_hw[0]):
        m = m.resize((target_hw[1], target_hw[0]), Image.NEAREST)
    return np.asarray(m, np.uint8) > 0


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_gray_png01(path, img01):
    img8 = (np.clip(img01, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(img8).convert("L").save(path)


# ------------------------
# Normalize for visualization
# ------------------------

def normalize_for_viz(err_map, valid_mask, p_low=1, p_high=99):
    vals = err_map[valid_mask]
    if vals.size == 0:
        return np.zeros_like(err_map), (0, 1)

    lo = np.percentile(vals, p_low)
    hi = np.percentile(vals, p_high)
    hi = max(hi, lo + 1e-6)

    out = (err_map - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    out[~valid_mask] = 0
    return out, (lo, hi)


# ------------------------
# Histogram (0–255 scale)
# ------------------------

def save_histogram(vals_255, out_path, title, bins=256):
    fig = Figure(figsize=(6, 4))
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)

    if vals_255.size > 0:
        ax.hist(vals_255, bins=bins, range=(0, 255))

    ax.set_xlim(0, 255)
    ax.set_xlabel("Absolute Error (0–255)")
    ax.set_ylabel("Pixel Count")
    ax.set_title(title)

    fig.tight_layout()
    canvas.draw()
    fig.savefig(out_path, dpi=150)


# ------------------------
# Montage 2x3
# ------------------------

def save_montage_2x3(paths_row1, paths_row2, out_path, title=None, pad=10, bg=(20, 20, 20)):
    ims1 = [Image.open(p).convert("RGB") for p in paths_row1]
    ims2 = [Image.open(p).convert("RGB") for p in paths_row2]

    w1, h1 = max(i.width for i in ims1), max(i.height for i in ims1)
    w2, h2 = max(i.width for i in ims2), max(i.height for i in ims2)

    ims1 = [i.resize((w1, h1), Image.BILINEAR) for i in ims1]
    ims2 = [i.resize((w2, h2), Image.BILINEAR) for i in ims2]

    row1_w = 3 * w1 + 2 * pad
    row2_w = 3 * w2 + 2 * pad
    out_w = max(row1_w, row2_w)
    title_h = 40 if title else 0
    out_h = title_h + h1 + pad + h2

    canvas = Image.new("RGB", (out_w, out_h), bg)
    draw = ImageDraw.Draw(canvas)

    if title:
        draw.text((10, 10), title, fill=(230, 230, 230))
        y0 = title_h
    else:
        y0 = 0

    x1 = (out_w - row1_w) // 2
    for i, im in enumerate(ims1):
        canvas.paste(im, (x1 + i * (w1 + pad), y0))

    y1 = y0 + h1 + pad
    x2 = (out_w - row2_w) // 2
    for i, im in enumerate(ims2):
        canvas.paste(im, (x2 + i * (w2 + pad), y1))

    canvas.save(out_path)


# ------------------------
# Main
# ------------------------

def process(folder_gt, folder_render, folder_mask, folder_out):
    ensure_dir(folder_out)

    channels = ["R", "G", "B"]
    idx = {"R": 0, "G": 1, "B": 2}

    out_err = os.path.join(folder_out, "error_norm")
    out_hist = os.path.join(folder_out, "hist")
    out_montage = os.path.join(folder_out, "montage")

    for c in channels:
        ensure_dir(os.path.join(out_err, c))
        ensure_dir(os.path.join(out_hist, c))
    ensure_dir(out_montage)

    gt_map = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(folder_gt)}
    rd_map = {os.path.splitext(os.path.basename(p))[0]: p for p in list_images(folder_render)}
    keys = sorted(set(gt_map) & set(rd_map))

    for name in keys:
        gt = read_rgb_float(gt_map[name])
        rd = read_rgb_float(rd_map[name])

        if gt.shape != rd.shape:
            rd = np.asarray(
                Image.fromarray((rd * 255).astype(np.uint8))
                .resize((gt.shape[1], gt.shape[0]), Image.BILINEAR),
                np.float32
            ) / 255.0

        H, W = gt.shape[:2]

        mask = None
        for ext in IMG_EXTS:
            p = os.path.join(folder_mask, name + ext)
            if os.path.exists(p):
                mask = p
                break

        valid = read_mask_bool(mask, (H, W)) if mask else np.ones((H, W), bool)

        err = np.abs(gt - rd)

        err_paths, hist_paths = [], []

        for c in channels:
            e = err[..., idx[c]]
            vals = e[valid]
            vals_255 = vals * 255.0

            norm, (lo, hi) = normalize_for_viz(e, valid)

            mn = float(vals_255.min()) if vals_255.size else 0
            mx = float(vals_255.max()) if vals_255.size else 0
            mean = float(vals_255.mean()) if vals_255.size else 0
            p95 = float(np.percentile(vals_255, 95)) if vals_255.size else 0

            err_path = os.path.join(out_err, c, f"{name}_err_norm_{c}.png")
            hist_path = os.path.join(out_hist, c, f"{name}_hist_{c}.png")

            save_gray_png01(err_path, norm)

            save_histogram(
                vals_255,
                hist_path,
                f"{name} | {c} channel\n"
                f"min={mn:.1f}, max={mx:.1f}, mean={mean:.2f}, p95={p95:.2f}"
            )

            err_paths.append(err_path)
            hist_paths.append(hist_path)

        save_montage_2x3(
            err_paths,
            hist_paths,
            os.path.join(out_montage, f"{name}_montage.png"),
            title=f"{name}   (top: normalized error | bottom: histogram 0–255)"
        )

    print("Done.")
    print("Results in:", folder_out)


if __name__ == "__main__":
    root_dir = "output/Deform3DGS-baseline/EndoNeRF/cutting_tissues_twice/video/ours_3000/"
    process(
        os.path.join(root_dir, "gt"),
        os.path.join(root_dir, "renders"),
        os.path.join(root_dir, "masks"),
        os.path.join(root_dir, "errors"),
    )
