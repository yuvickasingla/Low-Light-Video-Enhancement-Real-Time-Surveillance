import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.metrics import structural_similarity as ssim


def arithmetic_mean(img, k):
    return ndi.uniform_filter(img, size=k)


def geometric_mean(img, k, eps=1e-6):
    # product of a window ^ (1/n) == exp(mean(log(.))) -- avoids overflow
    log_img = np.log(img + eps)
    return np.exp(ndi.uniform_filter(log_img, size=k))


def harmonic_mean(img, k, eps=1e-6):
    # n / sum(1/z) == 1 / mean(1/z)
    return 1.0 / ndi.uniform_filter(1.0 / (img + eps), size=k)


def contraharmonic_mean(img, k, Q, eps=1e-6):
    z = img + eps
    num = ndi.uniform_filter(z ** (Q + 1), size=k)
    den = ndi.uniform_filter(z ** Q, size=k)
    return num / den


def median_filter(img_u8, k):
    return cv2.medianBlur(img_u8, k)


def min_filter(img, k):
    return ndi.minimum_filter(img, size=k)


def max_filter(img, k):
    return ndi.maximum_filter(img, size=k)


def midpoint_filter(img, k):
    return 0.5 * (min_filter(img, k) + max_filter(img, k))


def alpha_trimmed_mean(img, k, d):
    pad = k // 2
    padded = np.pad(img, pad, mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    windows = windows.reshape(img.shape[0], img.shape[1], k * k)
    sorted_w = np.sort(windows, axis=-1)
    trim = d // 2
    trimmed = sorted_w[..., trim: k * k - trim]
    return trimmed.mean(axis=-1)


def adaptive_local_noise_reduction(img, k, noise_var):
    local_mean = ndi.uniform_filter(img, size=k)
    local_sq_mean = ndi.uniform_filter(img ** 2, size=k)
    local_var = np.maximum(local_sq_mean - local_mean ** 2, 1e-8)
    ratio = np.minimum(noise_var / local_var, 1.0)
    return img - ratio * (img - local_mean)


def adaptive_median_filter(img_u8, smax=7):
    sizes = list(range(3, smax + 1, 2))
    mins = {s: ndi.minimum_filter(img_u8, size=s) for s in sizes}
    maxs = {s: ndi.maximum_filter(img_u8, size=s) for s in sizes}
    meds = {s: cv2.medianBlur(img_u8, s) for s in sizes}

    out = np.zeros_like(img_u8)
    determined = np.zeros(img_u8.shape, dtype=bool)
    img_i = img_u8.astype(np.int32)

    for s in sizes:
        zmin, zmed, zmax = mins[s].astype(np.int32), meds[s].astype(np.int32), maxs[s].astype(np.int32)
        A1, A2 = zmed - zmin, zmed - zmax
        level_a = (A1 > 0) & (A2 < 0)
        active = level_a & (~determined)

        B1, B2 = img_i - zmin, img_i - zmax
        level_b = (B1 > 0) & (B2 < 0)

        use_orig = active & level_b
        use_med = active & (~level_b)
        out[use_orig] = img_u8[use_orig]
        out[use_med] = meds[s][use_med]
        determined |= active

    # pixels that never satisfied level A even at Smax -> use median at Smax
    out[~determined] = meds[sizes[-1]][~determined]
    return out



def compute_metrics(orig_u8, filt_u8):
    mse = float(np.mean((orig_u8.astype(np.float64) - filt_u8.astype(np.float64)) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)
    ssim_val = ssim(orig_u8, filt_u8, data_range=255)
    return mse, psnr, ssim_val


def to_u8(img_float):
    return np.clip(img_float, 0, 255).astype(np.uint8)



WINDOW_SIZES = [3, 5]
NOISE_SETTINGS = [
    "gaussian_0_0.01", "gaussian_0_0.05",
    "rayleigh_0.1", "rayleigh_0.3",
    "erlang_2_5", "erlang_2_10",
    "exponential_1", "exponential_3",
    "uniform_-0.1_0.1", "uniform_-0.3_0.3",
    "salt_pepper_d0.05", "salt_pepper_d0.15",
]

PATCH = (slice(10, 50), slice(10, 50))


def estimate_noise_variance(noisy_u8, patch=PATCH):
    region = noisy_u8[patch].astype(np.float64)
    return float(np.var(region))


def build_filter_jobs(noisy_u8, noise_var):
    """Returns list of (label, filtered_uint8_image) for every filter/param combo."""
    noisy_f = noisy_u8.astype(np.float64)
    jobs = []

    for k in WINDOW_SIZES:
        jobs.append((f"arithmetic_mean_k{k}", to_u8(arithmetic_mean(noisy_f, k))))
        jobs.append((f"geometric_mean_k{k}", to_u8(geometric_mean(noisy_f, k))))
        jobs.append((f"harmonic_mean_k{k}", to_u8(harmonic_mean(noisy_f, k))))
        for Q in (1, -1):
            jobs.append((f"contraharmonic_k{k}_Q{Q}", to_u8(contraharmonic_mean(noisy_f, k, Q))))
        jobs.append((f"median_k{k}", median_filter(noisy_u8, k)))
        jobs.append((f"min_k{k}", to_u8(min_filter(noisy_f, k))))
        jobs.append((f"max_k{k}", to_u8(max_filter(noisy_f, k))))
        jobs.append((f"midpoint_k{k}", to_u8(midpoint_filter(noisy_f, k))))
        for d in (2, 6):
            jobs.append((f"alpha_trimmed_k{k}_d{d}", to_u8(alpha_trimmed_mean(noisy_f, k, d))))
        jobs.append((f"adaptive_local_noise_reduction_k{k}",
                      to_u8(adaptive_local_noise_reduction(noisy_f, k, noise_var))))

    jobs.append(("adaptive_median_Smax7", adaptive_median_filter(noisy_u8, smax=7)))

    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--a2_dir", type=str, default="../Assignment2/outputs/task_noise",
                         help="Path to Assignment 2's outputs/task_noise folder")
    parser.add_argument("--out_dir", type=str, default="outputs/task1_filters")
    args = parser.parse_args()

    a2_dir = Path(args.a2_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    for tag in ("smooth", "textured"):
        img_dir = a2_dir / tag
        orig_path = img_dir / "original.png"
        if not orig_path.exists():
            print(f"Skipping {tag}: {orig_path} not found")
            continue
        orig_u8 = cv2.imread(str(orig_path), cv2.IMREAD_GRAYSCALE)

        out_tag_dir = out_dir / tag
        out_tag_dir.mkdir(parents=True, exist_ok=True)

        for setting in NOISE_SETTINGS:
            noisy_path = img_dir / f"{setting}.png"
            if not noisy_path.exists():
                print(f"  Missing {noisy_path}, skipping")
                continue
            noisy_u8 = cv2.imread(str(noisy_path), cv2.IMREAD_GRAYSCALE)

            noise_var = estimate_noise_variance(noisy_u8)
            jobs = build_filter_jobs(noisy_u8, noise_var)

            setting_dir = out_tag_dir / setting
            setting_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(setting_dir / "noisy_input.png"), noisy_u8)

            for label, filt_u8 in jobs:
                cv2.imwrite(str(setting_dir / f"{label}.png"), filt_u8)
                mse, psnr, ssim_val = compute_metrics(orig_u8, filt_u8)
                all_rows.append([tag, setting, label, mse, psnr, ssim_val])
                print(f"[{tag}] {setting:20s} {label:35s} MSE={mse:8.2f} PSNR={psnr:6.2f} SSIM={ssim_val:.3f}")

    csv_path = out_dir / "restoration_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "noise_setting", "filter", "MSE", "PSNR_dB", "SSIM"])
        writer.writerows(all_rows)
    print(f"\nFull results written to {csv_path.resolve()}")

    # ---- consolidated per-noise-type pivot tables (Task 3 requirement) ----
    build_pivot_tables(all_rows, out_dir)


def build_pivot_tables(all_rows, out_dir):
    """One CSV per noise family, rows = filter, columns = (image, setting) PSNR."""
    import pandas as pd

    df = pd.DataFrame(all_rows, columns=["image", "noise_setting", "filter", "MSE", "PSNR_dB", "SSIM"])
    df["noise_family"] = df["noise_setting"].str.extract(r"^([a-z_]+?)(?:_-?\d|_d\d)")[0]
    df["noise_family"] = df["noise_family"].fillna(df["noise_setting"])

    pivot_dir = out_dir / "summary_tables"
    pivot_dir.mkdir(parents=True, exist_ok=True)

    for family, sub in df.groupby("noise_family"):
        pivot = sub.pivot_table(index="filter", columns=["image", "noise_setting"], values="PSNR_dB")
        pivot.to_csv(pivot_dir / f"{family}_PSNR_table.csv")

        pivot_mse = sub.pivot_table(index="filter", columns=["image", "noise_setting"], values="MSE")
        pivot_mse.to_csv(pivot_dir / f"{family}_MSE_table.csv")

    print(f"Per-noise-type summary tables written to {pivot_dir.resolve()}")


if __name__ == "__main__":
    main()
