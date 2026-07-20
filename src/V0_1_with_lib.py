"""
V0_1_with_lib.py — CPU Baseline (DUNG THU VIEN scipy)
======================================================

Phien ban nay su dung scipy.ndimage.gaussian_filter lam ham blur chinh.
Day la "reference implementation" chuan xac, duoc dung de:
  1. So sanh toc do voi V0_1_no_lib.py (separable 1D NumPy thu an)
  2. Lam ground-truth cho cac phien ban GPU (V1, V2, V3)
  3. Benchmark MAE: GPU output phai co MAE < 1e-3 so voi phien ban nay

Sau khi chay xong, script tu dong chay kem V0_1_no_lib de in bang so sanh.

Chay: python V0_1_with_lib.py
"""

import sys
import os
import time
import cProfile
import pstats
import io
import numpy as np

# Ep UTF-8 cho Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from scipy.ndimage import gaussian_filter   # Thu vien chinh

# OpenCV chi dung de load anh
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# =============================================================================
# PHAN 1: GAUSSIAN PYRAMID DUNG SCIPY
# =============================================================================

def make_gaussian_kernel_2d_ref(sigma, kernel_size=None):
    """
    Tao ma tran Gaussian 2D (tong = 1.0) — dung lam tham chieu.
    scipy.ndimage.gaussian_filter tu tinh kernel ben trong; ham nay chi
    de minh hoa gia tri kernel, KHONG dung trong qua trinh blur.
    """
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
    half = kernel_size // 2
    x = np.arange(-half, half + 1, dtype=np.float32)
    k1d = np.exp(-x * x / (2.0 * sigma * sigma))
    k1d /= k1d.sum()
    return np.outer(k1d, k1d).astype(np.float32)


def build_gaussian_pyramid_with_lib(image, num_octaves=4, num_scales=5, sigma_base=1.6):
    """
    Xay dung Gaussian Pyramid bang scipy.ndimage.gaussian_filter.

    Day la phien ban CHUAN XAC (reference) duoc dung lam:
      - Ket qua so sanh cho V0_1_no_lib
      - Ground truth de verify GPU kernels (MAE < 1e-3)

    Args:
        image      : anh grayscale float32 [0,1], shape (H, W)
        num_octaves: so octaves (mac dinh 4)
        num_scales : so scales moi octave (mac dinh 5)
        sigma_base : sigma co so (mac dinh 1.6)

    Returns:
        pyramid: list of list, pyramid[octave][scale] = ndarray (H', W')
    """
    pyramid = []
    current_img = image.astype(np.float32)
    k = 2.0 ** (1.0 / num_scales)

    for octave in range(num_octaves):
        octave_imgs = []

        for scale in range(num_scales):
            sigma = sigma_base * (k ** scale)
            # scipy.ndimage.gaussian_filter: dung FFT-based hoac separable IIR
            # tuy thuoc phien ban, ket qua chinh xac hon thu cai dat thu cong
            blurred = gaussian_filter(current_img, sigma=sigma)
            octave_imgs.append(blurred.astype(np.float32))

        pyramid.append(octave_imgs)

        # Downsample: lay anh giua octave, giam kich thuoc 2x
        mid_img = octave_imgs[num_scales // 2]
        current_img = np.ascontiguousarray(mid_img[::2, ::2])

    return pyramid


# =============================================================================
# PHAN 2: DoG (Difference of Gaussians)
# =============================================================================

def compute_dog_pyramid(gaussian_pyramid):
    """
    Tinh DoG Pyramid tu Gaussian Pyramid.
    DoG(i) = Gaussian(i+1) - Gaussian(i).
    """
    dog_pyramid = []
    for octave_imgs in gaussian_pyramid:
        dog_octave = [
            octave_imgs[i + 1] - octave_imgs[i]
            for i in range(len(octave_imgs) - 1)
        ]
        dog_pyramid.append(dog_octave)
    return dog_pyramid


# =============================================================================
# PHAN 3: LOAD ANH & SYNTHETIC IMAGE
# =============================================================================

def get_div2k_image_paths(dir_path):
    """Lay danh sach duong dan anh tu thu muc DIV2K."""
    if not os.path.exists(dir_path):
        return []
    candidates = sorted(
        name for name in os.listdir(dir_path)
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    )
    return [os.path.join(dir_path, name) for name in candidates]


def load_single_image(image_path):
    """Doc anh tu file → grayscale float32, range [0, 1]."""
    if not CV2_AVAILABLE:
        raise RuntimeError("OpenCV chua duoc cai dat — khong load duoc anh that.")
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Khong doc duoc anh: {image_path}")
    return img.astype(np.float32) / 255.0


def create_synthetic_div2k_image(seed=42):
    """Tao anh synthetic kich thuoc DIV2K trung binh (1356x2040)."""
    H, W = 1356, 2040
    rng = np.random.default_rng(seed)
    noise = rng.random((H, W)).astype(np.float32)
    grad_x = np.linspace(0, 1, W, dtype=np.float32)
    grad_y = np.linspace(0, 1, H, dtype=np.float32)
    gradient = np.outer(grad_y, grad_x)
    return (0.7 * noise + 0.3 * gradient).astype(np.float32)


# =============================================================================
# PHAN 4: BENCHMARK VA SO SANH HAI PHIEN BAN
# =============================================================================

def benchmark_single(build_fn, image, num_octaves, num_scales, n_runs=3, label=""):
    """
    Chay build_fn nhieu lan, tra ve thoi gian trung binh (giay).
    Warmup 1 lan truoc de tranh anh huong JIT / cache miss.
    """
    # Warmup
    build_fn(image, num_octaves, num_scales)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        pyramid = build_fn(image, num_octaves, num_scales)
        dog = compute_dog_pyramid(pyramid)
        times.append(time.perf_counter() - t0)

    avg = float(np.mean(times))
    return avg


def run_comparison(image, num_octaves, num_scales, n_runs=3):
    """
    Chay ca hai phien ban tren cung mot anh va in bang so sanh.
    Tra ve (time_nolib, time_withlib) tinh bang giay.
    """
    # Import phien ban no-lib
    try:
        from V0_1_no_lib import (
            build_gaussian_pyramid_no_lib,
            compute_dog_pyramid as compute_dog_no_lib,
        )
        nolib_available = True
    except ImportError:
        nolib_available = False
        print("  [WARN] Khong tim thay V0_1_no_lib.py — bo qua so sanh.")

    time_withlib = benchmark_single(
        build_gaussian_pyramid_with_lib, image, num_octaves, num_scales,
        n_runs=n_runs, label="With-Lib (scipy)")

    if nolib_available:
        time_nolib = benchmark_single(
            build_gaussian_pyramid_no_lib, image, num_octaves, num_scales,
            n_runs=n_runs, label="No-Lib (NumPy)")
        return time_nolib, time_withlib
    else:
        return None, time_withlib


def print_comparison_table(results):
    """
    In bang so sanh dang dep.
    results: list of dict {label, shape, time_nolib, time_withlib}
    """
    print()
    print("=" * 78)
    print(f"  {'BANG SO SANH: No-Lib (NumPy) vs With-Lib (scipy)'}")
    print("=" * 78)
    header = (
        f"  {'Anh / Kich thuoc':<28}  "
        f"{'No-Lib (ms)':>12}  "
        f"{'With-Lib (ms)':>14}  "
        f"{'Ty le (No/With)':>16}"
    )
    print(header)
    print("  " + "-" * 74)

    for r in results:
        if r["time_nolib"] is not None and r["time_withlib"] is not None:
            ratio = r["time_nolib"] / r["time_withlib"]
            ratio_str = f"{ratio:.1f}x"
        else:
            ratio_str = "N/A"

        nolib_str = f"{r['time_nolib']*1000:.1f}" if r["time_nolib"] is not None else "N/A"
        wlib_str  = f"{r['time_withlib']*1000:.1f}" if r["time_withlib"] is not None else "N/A"

        label = r.get("label", "?")[:28]
        print(f"  {label:<28}  {nolib_str:>12}  {wlib_str:>14}  {ratio_str:>16}")

    print("=" * 78)
    print("  Ghi chu: ty le No-Lib/With-Lib > 1 => scipy nhanh hon")
    print("           Muc tieu GPU: speedup 30-80x so voi With-Lib (scipy)")
    print("=" * 78)


# =============================================================================
# PHAN 5: MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  V0_1_with_lib.py -- Gaussian Pyramid (DUNG THU VIEN scipy)")
    print("=" * 65)
    print("  Thu vien: scipy.ndimage.gaussian_filter")
    print("  Vai tro : Reference baseline + so sanh voi V0_1_no_lib")
    print("=" * 65)

    NUM_OCTAVES  = 4
    NUM_SCALES   = 5
    LIMIT_IMAGES = 30
    N_RUNS       = 3   # So lan do de lay trung binh

    repo_root  = os.path.dirname(os.path.abspath(__file__))
    div2k_path = os.path.join(repo_root, "DIV2K_train_HR")

    image_paths = []
    if CV2_AVAILABLE and os.path.exists(div2k_path):
        image_paths = get_div2k_image_paths(div2k_path)

    # -------------------------------------------------------------------------
    # So sanh tren anh synthetic (luc nao cung chay duoc)
    # -------------------------------------------------------------------------
    print(f"\n[SYNTHETIC] Chay voi anh synthetic DIV2K (1356x2040)...")
    synth_img = create_synthetic_div2k_image(seed=42)
    H, W = synth_img.shape
    print(f"  Kich thuoc: {H}x{W} | Chay {N_RUNS} lan moi phien ban (+ 1 warmup)\n")

    time_nolib_s, time_wlib_s = run_comparison(
        synth_img, NUM_OCTAVES, NUM_SCALES, n_runs=N_RUNS)

    results_synth = [{
        "label": f"Synthetic ({H}x{W})",
        "time_nolib":  time_nolib_s,
        "time_withlib": time_wlib_s,
    }]

    # -------------------------------------------------------------------------
    # Benchmark tren anh DIV2K that (neu co)
    # -------------------------------------------------------------------------
    results_div2k = []
    if image_paths:
        print(f"\n[DIV2K] Tim thay {len(image_paths)} anh. "
              f"Benchmark {min(LIMIT_IMAGES, len(image_paths))} anh dau tien "
              f"({N_RUNS} runs + 1 warmup moi anh)...")

        selected = image_paths[:LIMIT_IMAGES]
        times_nolib_all  = []
        times_wlib_all   = []

        for idx, path in enumerate(selected):
            fname = os.path.basename(path)
            img   = load_single_image(path)
            Himg, Wimg = img.shape

            t_n, t_w = run_comparison(img, NUM_OCTAVES, NUM_SCALES, n_runs=N_RUNS)
            times_nolib_all.append(t_n)
            times_wlib_all.append(t_w)

            n_str = f"{t_n*1000:.1f}ms" if t_n else "N/A"
            w_str = f"{t_w*1000:.1f}ms" if t_w else "N/A"
            ratio = (t_n / t_w) if (t_n and t_w) else None
            r_str = f"{ratio:.1f}x" if ratio else "N/A"
            print(f"  [{idx+1:2d}/{len(selected)}] {fname} ({Himg}x{Wimg}): "
                  f"No-Lib={n_str}  scipy={w_str}  ratio={r_str}")

            results_div2k.append({
                "label"       : f"{fname[:20]}...",
                "time_nolib"  : t_n,
                "time_withlib": t_w,
            })

        # Them dong trung binh
        valid_n = [t for t in times_nolib_all if t is not None]
        valid_w = [t for t in times_wlib_all  if t is not None]
        avg_n = float(np.mean(valid_n)) if valid_n else None
        avg_w = float(np.mean(valid_w)) if valid_w else None
        results_div2k.append({
            "label"       : f"=== TRUNG BINH ({len(selected)} anh) ===",
            "time_nolib"  : avg_n,
            "time_withlib": avg_w,
        })

    # -------------------------------------------------------------------------
    # In bang so sanh cuoi cung
    # -------------------------------------------------------------------------
    print_comparison_table(results_synth + (results_div2k if results_div2k else []))

    # -------------------------------------------------------------------------
    # Benchmark chi With-Lib tren 1 anh full + cProfile
    # -------------------------------------------------------------------------
    print("\n[PROFILE] cProfile breakdown cua With-Lib tren anh synthetic:")
    pr = cProfile.Profile()
    pr.enable()
    build_gaussian_pyramid_with_lib(synth_img, NUM_OCTAVES, NUM_SCALES)
    pr.disable()
    sio = io.StringIO()
    ps = pstats.Stats(pr, stream=sio).sort_stats("cumulative")
    ps.print_stats(10)
    print(sio.getvalue())

    # -------------------------------------------------------------------------
    # Thong ke cau truc pyramid de tham khao
    # -------------------------------------------------------------------------
    print("[INFO] Cau truc pyramid (With-Lib, synthetic image):")
    pyramid = build_gaussian_pyramid_with_lib(synth_img, NUM_OCTAVES, NUM_SCALES)
    for o, octave in enumerate(pyramid):
        shapes = [f"{img.shape[0]}x{img.shape[1]}" for img in octave]
        print(f"  Octave {o}: " + "  ".join(shapes))

    print("\n[DONE] V0_1_with_lib.py hoan thanh.")


if __name__ == "__main__":
    main()
