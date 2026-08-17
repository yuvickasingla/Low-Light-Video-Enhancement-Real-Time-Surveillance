import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

OUT_DIR = Path("outputs/task_noise")


# ---------- noise generators (operate on float image in [0,1]) ----------

def add_gaussian(img, mean, var):
    sigma = np.sqrt(var)
    noise = np.random.normal(mean, sigma, img.shape)
    return img + noise, noise, ("gaussian", mean, var)


def add_rayleigh(img, scale):
    # Inverse-transform sampling: if U~Uniform(0,1),
    # Z = scale * sqrt(-2 ln(1-U)) ~ Rayleigh(scale)
    u = np.random.uniform(0, 1, img.shape)
    noise = scale * np.sqrt(-2 * np.log(1 - u + 1e-12))
    return img + noise, noise, ("rayleigh", scale)


def add_erlang(img, shape_b, rate_a):
    # numpy's gamma uses (shape, scale=1/rate)
    noise = np.random.gamma(shape=shape_b, scale=1.0 / rate_a, size=img.shape)
    return img + noise, noise, ("erlang", shape_b, rate_a)


def add_exponential(img, rate_a):
    noise = np.random.exponential(scale=1.0 / rate_a, size=img.shape)
    return img + noise, noise, ("exponential", rate_a)


def add_uniform(img, a, b):
    noise = np.random.uniform(a, b, img.shape)
    return img + noise, noise, ("uniform", a, b)


def add_salt_pepper(img_uint8, density):
    noisy = img_uint8.copy()
    mask = np.random.rand(*img_uint8.shape)
    noisy[mask < density / 2] = 0
    noisy[(mask >= density / 2) & (mask < density)] = 255
    return noisy, ("salt_pepper", density)


# ---------- theoretical PDFs, for overlay on the extracted-noise histogram ----------

def theoretical_pdf(kind, params, z):
    if kind == "gaussian":
        mean, var = params
        return stats.norm.pdf(z, loc=mean, scale=np.sqrt(var))
    if kind == "rayleigh":
        (scale,) = params
        return stats.rayleigh.pdf(z, scale=scale)
    if kind == "erlang":
        shape_b, rate_a = params
        return stats.gamma.pdf(z, a=shape_b, scale=1.0 / rate_a)
    if kind == "exponential":
        (rate_a,) = params
        return stats.expon.pdf(z, scale=1.0 / rate_a)
    if kind == "uniform":
        a, b = params
        return stats.uniform.pdf(z, loc=a, scale=b - a)
    raise ValueError(kind)


PDF_FORMULA = {
    "gaussian": "p(z) = 1/(sqrt(2*pi)*sigma) * exp(-(z-mean)^2 / (2*sigma^2))",
    "rayleigh": "p(z) = (z/scale^2) * exp(-z^2 / (2*scale^2)),  z >= 0",
    "erlang":   "p(z) = a^b * z^(b-1) * exp(-a*z) / (b-1)!,  z >= 0",
    "exponential": "p(z) = a * exp(-a*z),  z >= 0",
    "uniform":  "p(z) = 1/(b-a) for a <= z <= b, else 0",
    "salt_pepper": "p(z) = P_pepper*delta(0) + P_salt*delta(255) (impulses only)",
}


# ---------- helpers ----------

def load_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def mse_psnr(orig_u8, noisy_u8):
    mse = float(np.mean((orig_u8.astype(np.float64) - noisy_u8.astype(np.float64)) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)
    return mse, psnr


def save_hist(values, title, path, bins=256, rng=(0, 255)):
    plt.figure(figsize=(4, 3))
    plt.hist(values.ravel(), bins=bins, range=rng, color="steelblue")
    plt.title(title, fontsize=9)
    plt.xlabel("Gray level")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def save_noise_hist_with_pdf(noise_patch_float, kind, params, title, path):
    plt.figure(figsize=(4, 3))
    plt.hist(noise_patch_float.ravel(), bins=60, density=True,
              color="lightcoral", label="extracted noise (patch)")
    if kind != "salt_pepper":
        lo, hi = noise_patch_float.min(), noise_patch_float.max()
        z = np.linspace(lo, hi, 300)
        pdf = theoretical_pdf(kind, params, z)
        plt.plot(z, pdf, "k-", lw=2, label="theoretical PDF")
    plt.title(title, fontsize=9)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


# ---------- main pipeline ----------

def process_image(img_path, tag, patch=(slice(10, 50), slice(10, 50))):
    img_u8 = load_gray(img_path)
    img_f = img_u8.astype(np.float64) / 255.0
    results = []

    out_dir = OUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    save_hist(img_u8, f"{tag}: original histogram", out_dir / "original_hist.png")
    cv2.imwrite(str(out_dir / "original.png"), img_u8)

    configs = [
        ("gaussian", lambda: add_gaussian(img_f, 0, 0.01)),
        ("gaussian", lambda: add_gaussian(img_f, 0, 0.05)),
        ("rayleigh", lambda: add_rayleigh(img_f, 0.1)),
        ("rayleigh", lambda: add_rayleigh(img_f, 0.3)),
        ("erlang", lambda: add_erlang(img_f, 2, 5)),
        ("erlang", lambda: add_erlang(img_f, 2, 10)),
        ("exponential", lambda: add_exponential(img_f, 1)),
        ("exponential", lambda: add_exponential(img_f, 3)),
        ("uniform", lambda: add_uniform(img_f, -0.1, 0.1)),
        ("uniform", lambda: add_uniform(img_f, -0.3, 0.3)),
    ]

    for kind, fn in configs:
        noisy_f, noise_f, meta = fn()
        noisy_f = np.clip(noisy_f, 0, 1)
        noisy_u8 = (noisy_f * 255).astype(np.uint8)

        label = "_".join(str(m) for m in meta)
        cv2.imwrite(str(out_dir / f"{label}.png"), noisy_u8)
        save_hist(noisy_u8, f"{tag}: {label}", out_dir / f"{label}_hist.png")

        patch_noise = (noisy_u8[patch].astype(np.float64) / 255.0) - img_f[patch]
        params = meta[1:]
        save_noise_hist_with_pdf(patch_noise, kind, params,
                                  f"{tag}: {label} noise vs theory",
                                  out_dir / f"{label}_noise_vs_pdf.png")

        mse, psnr = mse_psnr(img_u8, noisy_u8)
        results.append((tag, label, kind, PDF_FORMULA[kind], mse, psnr))
        print(f"[{tag}] {label:30s} MSE={mse:8.2f}  PSNR={psnr:6.2f} dB")

    # salt-and-pepper (integer domain, handled separately)
    for density in (0.05, 0.15):
        noisy_u8, meta = add_salt_pepper(img_u8, density)
        label = f"salt_pepper_d{density}"
        cv2.imwrite(str(out_dir / f"{label}.png"), noisy_u8)
        save_hist(noisy_u8, f"{tag}: {label}", out_dir / f"{label}_hist.png")

        patch_noise = noisy_u8[patch].astype(np.float64) - img_u8[patch].astype(np.float64)
        save_noise_hist_with_pdf(patch_noise, "salt_pepper", (density,),
                                  f"{tag}: {label} noise", out_dir / f"{label}_noise_vs_pdf.png")

        mse, psnr = mse_psnr(img_u8, noisy_u8)
        results.append((tag, label, "salt_pepper", PDF_FORMULA["salt_pepper"], mse, psnr))
        print(f"[{tag}] {label:30s} MSE={mse:8.2f}  PSNR={psnr:6.2f} dB")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smooth", required=True, help="Path to a low-detail/smooth image")
    parser.add_argument("--textured", required=True, help="Path to a high-detail/textured image")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    all_results += process_image(args.smooth, "smooth")
    all_results += process_image(args.textured, "textured")

    import csv
    csv_path = OUT_DIR / "mse_psnr_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "noise_setting", "noise_type", "pdf_formula", "MSE", "PSNR_dB"])
        writer.writerows(all_results)

    print(f"\nSummary table written to {csv_path.resolve()}")
    print(f"All images/histograms written under {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
