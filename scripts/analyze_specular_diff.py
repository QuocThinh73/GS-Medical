#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "So sánh ảnh gốc và ảnh inpaint theo từng kênh màu, trên toàn ảnh / vùng specular / vùng không specular."
        )
    )
    parser.add_argument("--orig_dir", type=str, default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF/cutting_tissues_twice/images", help="Folder ảnh gốc")
    parser.add_argument("--inpaint_dir", type=str, default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF/cutting_tissues_twice/inpaint_images", help="Folder ảnh inpaint specular")
    parser.add_argument("--mask_dir", type=str, default="/media/dial2/Ubuntu Volume/dataset/EndoNeRF/cutting_tissues_twice/specular_masks", help="Folder mask nhị phân: trắng = specular, đen = background")
    parser.add_argument("--out_dir", type=str, default="./output", help="Folder lưu kết quả")
    parser.add_argument(
        "--color_spaces",
        nargs="+",
        default=["rgb", "hsv"],
        choices=["rgb", "hsv", "lab", "ycrcb", "gray"],
        help="Các không gian màu cần phân tích",
    )
    parser.add_argument(
        "--mask_threshold",
        type=int,
        default=127,
        help="Ngưỡng nhị phân cho mask nếu mask không hoàn toàn 0/255",
    )
    parser.add_argument(
        "--save_figures",
        action="store_true",
        help="Lưu ảnh trực quan hóa cho từng ảnh",
    )
    parser.add_argument(
        "--max_figures",
        type=int,
        default=-1,
        help="Số lượng ảnh tối đa cần lưu figure. -1 = lưu tất cả",
    )
    parser.add_argument(
        "--resize_mask_if_needed",
        action="store_true",
        help="Nếu mask khác kích thước ảnh thì resize mask bằng nearest neighbor",
    )
    return parser.parse_args()


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def list_image_files(folder: Path) -> Dict[str, Path]:
    files = {}
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            files[p.name] = p
    return files


def read_rgb_image(path: Path) -> np.ndarray:
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Không đọc được mask: {path}")
    return mask


def prepare_mask(mask: np.ndarray, target_hw: Tuple[int, int], threshold: int, resize_if_needed: bool) -> np.ndarray:
    h, w = target_hw
    if mask.shape != (h, w):
        if not resize_if_needed:
            raise ValueError(
                f"Mask có kích thước {mask.shape} khác ảnh {(h, w)}. Dùng --resize_mask_if_needed nếu muốn resize."
            )
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return (mask > threshold)


def convert_color_space(img_rgb_u8: np.ndarray, space: str) -> Tuple[np.ndarray, List[str]]:
    if space == "rgb":
        return img_rgb_u8.astype(np.float32), ["R", "G", "B"]
    if space == "hsv":
        hsv = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
        return hsv, ["H", "S", "V"]
    if space == "lab":
        lab = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2LAB).astype(np.float32)
        return lab, ["L", "A", "B"]
    if space == "ycrcb":
        ycrcb = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        return ycrcb, ["Y", "Cr", "Cb"]
    if space == "gray":
        gray = cv2.cvtColor(img_rgb_u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
        return gray[..., None], ["Gray"]
    raise ValueError(f"Không hỗ trợ color space: {space}")


def circular_diff_hue(h1: np.ndarray, h2: np.ndarray) -> np.ndarray:
    # OpenCV hue nằm trong [0, 179]
    d = h1 - h2
    return ((d + 90.0) % 180.0) - 90.0


def masked_values(arr2d: np.ndarray, region: np.ndarray) -> np.ndarray:
    vals = arr2d[region]
    return vals.astype(np.float32)


def compute_stats(values: np.ndarray) -> Dict[str, float]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "mean_abs": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "rmse": None,
        }
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "mean_abs": float(np.mean(np.abs(values))),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "rmse": float(np.sqrt(np.mean(values ** 2))),
    }


def build_region_masks(spec_mask: np.ndarray) -> Dict[str, np.ndarray]:
    return {
        "full": np.ones_like(spec_mask, dtype=bool),
        "specular": spec_mask,
        "non_specular": ~spec_mask,
    }


def analyze_one_image(
    orig_rgb: np.ndarray,
    inpaint_rgb: np.ndarray,
    spec_mask: np.ndarray,
    color_spaces: List[str],
) -> Dict[str, Dict]:
    results = {}
    region_masks = build_region_masks(spec_mask)

    for color_space in color_spaces:
        orig_cs, channel_names = convert_color_space(orig_rgb, color_space)
        inp_cs, _ = convert_color_space(inpaint_rgb, color_space)

        diff = orig_cs - inp_cs
        if color_space == "hsv":
            diff[..., 0] = circular_diff_hue(orig_cs[..., 0], inp_cs[..., 0])

        space_result = {
            "channel_names": channel_names,
            "regions": {},
        }

        for region_name, region_mask in region_masks.items():
            region_result = {}
            for c, cname in enumerate(channel_names):
                vals = masked_values(diff[..., c], region_mask)
                region_result[cname] = compute_stats(vals)
            space_result["regions"][region_name] = region_result

        results[color_space] = space_result
    return results


def make_overlay(orig_rgb: np.ndarray, spec_mask: np.ndarray) -> np.ndarray:
    overlay = orig_rgb.astype(np.float32).copy()
    overlay[spec_mask] = 0.6 * overlay[spec_mask] + 0.4 * np.array([255, 0, 0], dtype=np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def normalize_diff_map(diff2d: np.ndarray) -> np.ndarray:
    max_abs = np.max(np.abs(diff2d))
    if max_abs < 1e-8:
        return np.zeros_like(diff2d)
    return diff2d / max_abs


def save_visualization(
    save_path: Path,
    image_name: str,
    orig_rgb: np.ndarray,
    inpaint_rgb: np.ndarray,
    spec_mask: np.ndarray,
    color_spaces: List[str],
):
    nrows = len(color_spaces)
    ncols = 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4.5 * ncols, 4.2 * nrows))
    if nrows == 1:
        axes = np.expand_dims(axes, axis=0)

    overlay = make_overlay(orig_rgb, spec_mask)

    for i, color_space in enumerate(color_spaces):
        orig_cs, channel_names = convert_color_space(orig_rgb, color_space)
        inp_cs, _ = convert_color_space(inpaint_rgb, color_space)
        diff = orig_cs - inp_cs
        if color_space == "hsv":
            diff[..., 0] = circular_diff_hue(orig_cs[..., 0], inp_cs[..., 0])

        axes[i, 0].imshow(orig_rgb)
        axes[i, 0].set_title(f"{image_name}\nOriginal")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(inpaint_rgb)
        axes[i, 1].set_title("Inpaint")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(overlay)
        axes[i, 2].set_title("Mask overlay (đỏ = specular)")
        axes[i, 2].axis("off")

        show_idx = 0
        norm_map = normalize_diff_map(diff[..., show_idx])
        im1 = axes[i, 3].imshow(norm_map, cmap="bwr", vmin=-1, vmax=1)
        axes[i, 3].set_title(f"{color_space.upper()} diff: {channel_names[show_idx]}")
        axes[i, 3].axis("off")
        fig.colorbar(im1, ax=axes[i, 3], fraction=0.046, pad=0.04)

        vals_full = diff[..., show_idx].reshape(-1)
        vals_spec = diff[..., show_idx][spec_mask]
        vals_non = diff[..., show_idx][~spec_mask]
        axes[i, 4].hist(vals_full, bins=100, alpha=0.5, label="full")
        if vals_spec.size > 0:
            axes[i, 4].hist(vals_spec, bins=100, alpha=0.5, label="specular")
        if vals_non.size > 0:
            axes[i, 4].hist(vals_non, bins=100, alpha=0.5, label="non_specular")
        axes[i, 4].set_title(f"Histogram diff: {channel_names[show_idx]}")
        axes[i, 4].legend()

    plt.tight_layout()
    fig.savefig(save_path, dpi=180)
    plt.close(fig)


def summarize_dataset(all_results: Dict[str, Dict]) -> Dict[str, Dict]:
    accumulator = {}

    for _, image_result in all_results.items():
        for color_space, space_result in image_result.items():
            accumulator.setdefault(color_space, {})
            channel_names = space_result["channel_names"]
            for region_name, region_result in space_result["regions"].items():
                accumulator[color_space].setdefault(region_name, {})
                for cname in channel_names:
                    accumulator[color_space][region_name].setdefault(cname, [])
                    stat = region_result[cname]
                    if stat["count"] > 0:
                        accumulator[color_space][region_name][cname].append(stat)

    summary = {}
    for color_space, color_data in accumulator.items():
        summary[color_space] = {}
        for region_name, region_data in color_data.items():
            summary[color_space][region_name] = {}
            for cname, stats_list in region_data.items():
                total_count = sum(s["count"] for s in stats_list)
                if total_count == 0:
                    summary[color_space][region_name][cname] = {
                        "images": 0,
                        "pixels": 0,
                        "weighted_mean": None,
                        "weighted_mean_abs": None,
                        "weighted_rmse": None,
                        "avg_std_per_image": None,
                    }
                    continue

                weighted_mean = sum(s["mean"] * s["count"] for s in stats_list) / total_count
                weighted_mean_abs = sum(s["mean_abs"] * s["count"] for s in stats_list) / total_count
                weighted_mse = sum((s["rmse"] ** 2) * s["count"] for s in stats_list) / total_count
                avg_std = sum(s["std"] for s in stats_list) / len(stats_list)

                summary[color_space][region_name][cname] = {
                    "images": len(stats_list),
                    "pixels": int(total_count),
                    "weighted_mean": float(weighted_mean),
                    "weighted_mean_abs": float(weighted_mean_abs),
                    "weighted_rmse": float(np.sqrt(weighted_mse)),
                    "avg_std_per_image": float(avg_std),
                }
    return summary


def print_console_summary(summary: Dict[str, Dict]):
    print("\n" + "=" * 90)
    print("TỔNG KẾT TOÀN DATASET")
    print("=" * 90)
    for color_space, color_data in summary.items():
        print(f"\n[{color_space.upper()}]")
        for region_name, region_data in color_data.items():
            print(f"  - Region: {region_name}")
            for cname, stat in region_data.items():
                print(
                    f"    {cname:>6} | images={stat['images']:4d} | pixels={stat['pixels']:10d} | "
                    f"mean={stat['weighted_mean']:.4f} | mean_abs={stat['weighted_mean_abs']:.4f} | "
                    f"rmse={stat['weighted_rmse']:.4f} | avg_std={stat['avg_std_per_image']:.4f}"
                )


def main():
    args = parse_args()

    orig_dir = Path(args.orig_dir)
    inpaint_dir = Path(args.inpaint_dir)
    mask_dir = Path(args.mask_dir)
    out_dir = Path(args.out_dir)

    ensure_dir(out_dir)
    figures_dir = out_dir / "figures"
    per_image_dir = out_dir / "per_image_json"
    ensure_dir(figures_dir)
    ensure_dir(per_image_dir)

    orig_files = list_image_files(orig_dir)
    inpaint_files = list_image_files(inpaint_dir)
    mask_files = list_image_files(mask_dir)

    common_names = sorted(set(orig_files) & set(inpaint_files) & set(mask_files))
    if not common_names:
        raise ValueError("Không có filename chung giữa 3 folder.")

    missing_orig = sorted((set(inpaint_files) & set(mask_files)) - set(orig_files))
    missing_inpaint = sorted((set(orig_files) & set(mask_files)) - set(inpaint_files))
    missing_mask = sorted((set(orig_files) & set(inpaint_files)) - set(mask_files))

    if missing_orig:
        print(f"[Cảnh báo] Thiếu ở orig_dir: {missing_orig[:10]}{' ...' if len(missing_orig) > 10 else ''}")
    if missing_inpaint:
        print(f"[Cảnh báo] Thiếu ở inpaint_dir: {missing_inpaint[:10]}{' ...' if len(missing_inpaint) > 10 else ''}")
    if missing_mask:
        print(f"[Cảnh báo] Thiếu ở mask_dir: {missing_mask[:10]}{' ...' if len(missing_mask) > 10 else ''}")

    print(f"Tìm thấy {len(common_names)} ảnh khớp tên giữa 3 folder.")

    dataset_results = {}
    saved_fig_count = 0

    for idx, name in enumerate(common_names, start=1):
        print(f"[{idx}/{len(common_names)}] Đang xử lý: {name}")
        orig_rgb = read_rgb_image(orig_files[name])
        inpaint_rgb = read_rgb_image(inpaint_files[name])
        if orig_rgb.shape != inpaint_rgb.shape:
            raise ValueError(
                f"Ảnh gốc và inpaint khác kích thước ở file {name}: {orig_rgb.shape} vs {inpaint_rgb.shape}"
            )

        raw_mask = read_mask(mask_files[name])
        spec_mask = prepare_mask(
            raw_mask,
            target_hw=orig_rgb.shape[:2],
            threshold=args.mask_threshold,
            resize_if_needed=args.resize_mask_if_needed,
        )

        image_result = analyze_one_image(
            orig_rgb=orig_rgb,
            inpaint_rgb=inpaint_rgb,
            spec_mask=spec_mask,
            color_spaces=args.color_spaces,
        )
        dataset_results[name] = image_result

        with open(per_image_dir / f"{Path(name).stem}.json", "w", encoding="utf-8") as f:
            json.dump(image_result, f, indent=2, ensure_ascii=False)

        should_save_figure = args.save_figures and (args.max_figures < 0 or saved_fig_count < args.max_figures)
        if should_save_figure:
            save_visualization(
                save_path=figures_dir / f"{Path(name).stem}_viz.png",
                image_name=name,
                orig_rgb=orig_rgb,
                inpaint_rgb=inpaint_rgb,
                spec_mask=spec_mask,
                color_spaces=args.color_spaces,
            )
            saved_fig_count += 1

    summary = summarize_dataset(dataset_results)
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(out_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(dataset_results, f, indent=2, ensure_ascii=False)

    print_console_summary(summary)
    print(f"\nĐã lưu kết quả vào: {out_dir}")
    print(f"- Summary toàn bộ dataset: {out_dir / 'summary.json'}")
    print(f"- Kết quả từng ảnh:        {per_image_dir}")
    if args.save_figures:
        print(f"- Hình trực quan hóa:      {figures_dir}")


if __name__ == "__main__":
    main()
