#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict
from PIL import Image


def list_images(folder: Path):
    """List only .png images in the given folder (non-recursive)."""
    return [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".png"]


def compute_average_brightness(image_path: Path) -> float:
    """Average grayscale brightness in [0, 255]."""
    with Image.open(image_path) as im:
        gray = im.convert("L")
        arr = np.asarray(gray, dtype=np.float32)
    return float(arr.mean())


def classify_brightness(la: float, La: float, t1: float) -> str:
    """
    T1 = (la - La) / la

    high   if T1 >  t1
    medium if -t1 < T1 < t1
    low    if T1 < -t1
    """
    if la <= 1e-12:
        return "low"

    T1 = (la - La) / la

    if T1 > t1:
        return "high"
    elif T1 < -t1:
        return "low"
    else:
        return "medium"


def main():
    parser = argparse.ArgumentParser(
        description="Brightness classification for .png images (folder/images/*.png)."
    )
    parser.add_argument("root", type=str, help="Root folder (expects subfolder: images/).")
    parser.add_argument("--La", type=float, default=112.0)
    parser.add_argument("--t1", type=float, default=0.3)

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    folder = root / "images"

    if not folder.exists() or not folder.is_dir():
        raise SystemExit(f"Error: images folder not found: {folder}")

    images = sorted(list_images(folder))

    results: Dict[str, str] = {}

    for img_path in images:
        la = compute_average_brightness(img_path)
        label = classify_brightness(la, args.La, args.t1)
        results[img_path.name] = label

    out_path = root / "brightness.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
