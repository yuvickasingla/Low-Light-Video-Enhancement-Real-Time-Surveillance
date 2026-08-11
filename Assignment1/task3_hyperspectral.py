import os
from pathlib import Path

import numpy as np
import requests
import scipy.io as sio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_URL = "https://huggingface.co/datasets/danaroth/indian_pines/resolve/main/Indian_pines_corrected.mat"
GT_URL = "https://huggingface.co/datasets/danaroth/indian_pines/resolve/main/Indian_pines_gt.mat"

DATA_DIR = Path("data/hyperspectral")
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / "task3"


def download(url: str, dest: Path):
    if dest.exists():
        print(f"  Already downloaded: {dest}")
        return
    print(f"  Downloading {url} ...")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cube_path = DATA_DIR / "Indian_pines_corrected.mat"
    gt_path = DATA_DIR / "Indian_pines_gt.mat"

    print("== Downloading Indian Pines hyperspectral scene ==")
    download(DATA_URL, cube_path)
    download(GT_URL, gt_path)

    cube = sio.loadmat(cube_path)["indian_pines_corrected"]  # (H, W, Bands)
    gt = sio.loadmat(gt_path)["indian_pines_gt"]

    h, w, bands = cube.shape
    print("\n== Report fields ==")
    print("Dataset / sensor: AVIRIS airborne imaging spectrometer, "
          "Indian Pines test site, NW Indiana, USA")
    print(f"Spatial dimensions: {w} x {h}")
    print(f"Number of bands: {bands} (20 water-absorption bands already removed "
          f"from the original 224)")
    print("Approximate wavelength range: ~400 nm - 2500 nm (visible to SWIR)")

    # --- False-colour composite (pick 3 representative bands) ---
    r_band, g_band, b_band = bands // 4 * 3, bands // 2, bands // 8
    false_color = np.stack(
        [cube[:, :, r_band], cube[:, :, g_band], cube[:, :, b_band]], axis=-1
    ).astype(np.float32)
    false_color -= false_color.min()
    false_color /= (false_color.max() + 1e-6)

    plt.figure(figsize=(5, 5))
    plt.imshow(false_color)
    plt.title(f"False-colour composite (bands {r_band},{g_band},{b_band})")
    plt.axis("off")
    plt.savefig(OUT_DIR / "false_color_composite.png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nSaved false-colour composite -> {OUT_DIR/'false_color_composite.png'}")

    # --- Single pixel spectrum ---
    py, px = h // 2, w // 2
    spectrum = cube[py, px, :]

    plt.figure(figsize=(6, 3))
    plt.plot(spectrum)
    plt.xlabel("Band index")
    plt.ylabel("Reflectance (raw digital number)")
    plt.title(f"Spectrum at pixel ({px},{py})")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "pixel_spectrum.png", dpi=150)
    plt.close()
    print(f"Saved pixel spectrum plot -> {OUT_DIR/'pixel_spectrum.png'}")
    print(f"\nPixel ({px},{py}) spectrum, first 10 bands: {spectrum[:10]}")

    print("\n== Discussion (paste into your report, edit freely) ==")
    print(
        "Two objects that look identical in an RGB image (same colour to the "
        "naked eye/camera) can have very different reflectance curves across "
        "the 200 bands here — e.g. healthy vs. stressed vegetation reflect "
        "similarly in visible light but diverge sharply in the near-infrared "
        "bands. This is the basis for applications like crop health monitoring, "
        "mineral identification, and food-quality inspection, where hyperspectral "
        "imaging distinguishes materials that RGB cameras cannot."
    )


if __name__ == "__main__":
    main()
