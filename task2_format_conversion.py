import argparse
from pathlib import Path

import cv2
import numpy as np


def convert_and_report(image_path: str, out_dir: str = "outputs/task2"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # OpenCV reads as BGR by default
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    h, w, c = bgr.shape
    print(f"Original image: {image_path}")
    print(f"  Dimensions: {w} x {h}, channels: {c}, dtype: {bgr.dtype}")
    print(f"  Value range: [{bgr.min()}, {bgr.max()}]")

    # --- Original (as loaded, BGR) ---
    cv2.imwrite(str(out_dir / "original.png"), bgr)

    # --- RGB ---
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cv2.imwrite(str(out_dir / "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    print(f"\nRGB: shape={rgb.shape}, dtype={rgb.dtype}, range=[{rgb.min()},{rgb.max()}]")

    # --- YCbCr ---
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    cv2.imwrite(str(out_dir / "ycbcr.png"), ycrcb)
    print(f"YCbCr: shape={ycrcb.shape}, dtype={ycrcb.dtype}, "
          f"Y range=[{y.min()},{y.max()}], Cb range=[{cb.min()},{cb.max()}], "
          f"Cr range=[{cr.min()},{cr.max()}]")
    print("  Conversion formula used (ITU-R BT.601, as OpenCV implements it):")
    print("    Y  = 0.299R + 0.587G + 0.114B")
    print("    Cr = (R - Y) * 0.713 + 128")
    print("    Cb = (B - Y) * 0.564 + 128")

    # --- Grayscale ---
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(out_dir / "grayscale.png"), gray)
    print(f"\nGrayscale: shape={gray.shape}, dtype={gray.dtype}, "
          f"range=[{gray.min()},{gray.max()}]")
    print("  Formula: Y = 0.299R + 0.587G + 0.114B (same luma formula as above)")

    # --- Binary (Otsu automatic thresholding) ---
    otsu_thresh, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    cv2.imwrite(str(out_dir / "binary_otsu.png"), binary)
    print(f"\nBinary (Otsu): threshold chosen automatically = {otsu_thresh:.1f}")
    print(f"  Pixel values present: {sorted(np.unique(binary).tolist())}")

    print(f"\nAll outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                         help="Path to one representative ExDark image")
    parser.add_argument("--out_dir", type=str, default="outputs/task2")
    args = parser.parse_args()
    convert_and_report(args.image, args.out_dir)
