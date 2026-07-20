"""
V0_1_no_lib.py — CPU Baseline (KHÔNG DÙNG THƯ VIỆN xử lý ảnh)
===============================================================

Phiên bản này tự cài đặt HOÀN TOÀN bằng NumPy thuần:
  - Tự tạo Gaussian kernel 1D
  - Tự thực hiện Gaussian blur bằng separable 1D convolution
  - Tự thực hiện reflect padding, downsample 2×, và DoG

KHÔNG dùng: scipy, cv2.filter2D, hay bất kỳ hàm convolution có sẵn nào.

Mục đích: làm baseline "từ gốc" để so sánh với V0_1_with_lib.py
và làm điểm xuất phát cho các phiên bản GPU sau này.

Chạy: python V0_1_no_lib.py
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

# OpenCV chi dung de load anh (khong dung cho xu ly)
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


# =============================================================================
# PHAN 1: TU CAI DAT GAUSSIAN KERNEL & BLUR (khong dung thu vien)
# =============================================================================

def make_gaussian_kernel_1d(sigma, kernel_size=None):
    """
    Tao Gaussian kernel 1D (tong = 1.0) — tu cai dat bang NumPy.

    Cong thuc: G(x) = exp(-x^2 / (2*sigma^2))
    Sau do normalize de tong = 1.
    """
    if kernel_size is None:
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1

    half = kernel_size // 2
    x = np.arange(-half, half + 1, dtype=np.float64)
    kernel = np.exp(-x * x / (2.0 * sigma * sigma))
    kernel /= kernel.sum()
    return kernel.astype(np.float32)


def convolve_1d_horizontal(image, kernel):
    """
    Tu cai dat convolution 1D theo chieu ngang (doc theo cols).
    Dung vectorized NumPy — khong dung ham conv san co.

    Padding kieu reflect duoc thuc hien noi tuyen.
    """
    H, W = image.shape
    k = len(kernel)
    half = k // 2

    # Reflect pad theo chieu ngang
    left = image[:, half:0:-1]            # mirror ben trai
    right = image[:, -2:-half - 2:-1]     # mirror ben phai
    padded = np.concatenate([left, image, right], axis=1)

    out = np.zeros((H, W), dtype=np.float32)
    for i, w in enumerate(kernel):
        # Moi phan tu kernel nhan voi mot dai cot tuong ung — vectorized
        out += padded[:, i:i + W] * w
    return out


def convolve_1d_vertical(image, kernel):
    """
    Tu cai dat convolution 1D theo chieu doc (doc theo rows).
    Padding kieu reflect duoc thuc hien noi tuyen.
    """
    H, W = image.shape
    k = len(kernel)
    half = k // 2

    # Reflect pad theo chieu doc
    top = image[half:0:-1, :]             # mirror phia tren
    bottom = image[-2:-half - 2:-1, :]   # mirror phia duoi
    padded = np.concatenate([top, image, bottom], axis=0)

    out = np.zeros((H, W), dtype=np.float32)
    for i, w in enumerate(kernel):
        out += padded[i:i + H, :] * w
    return out


def gaussian_blur_separable(image, sigma, kernel_size=None):
    """
    Gaussian blur 2D bang ky thuat separable filter — tu cai dat.

    Thay vi tich chap voi kernel 2D (O(k^2) moi pixel), ap dung 2 lan 1D:
      Buoc 1: Conv 1D ngang → anh trung gian
      Buoc 2: Conv 1D doc   → anh ket qua

    Do phuc tap: O(k) thay vi O(k^2) → nhanh hon dang ke.
    """
    k1d = make_gaussian_kernel_1d(sigma, kernel_size)
    temp = convolve_1d_horizontal(image, k1d)
    out  = convolve_1d_vertical(temp, k1d)
    return out


# =============================================================================
# PHAN 2: TU CAI DAT GAUSSIAN PYRAMID
# =============================================================================

def downsample_2x(image):
    """
    Giam kich thuoc anh 2x bang cach lay moi pixel thu 2 (nearest neighbor).
    Khong dung ham resize cua thu vien.
    """
    return np.ascontiguousarray(image[::2, ::2])


def build_gaussian_pyramid_no_lib(image, num_octaves=4, num_scales=5, sigma_base=1.6):
    """
    Xay dung Gaussian Pyramid — TU CAI DAT hoan toan bang NumPy.

    Cau truc:
      - num_octaves octaves, moi octave num_scales scales
      - Moi scale: blur anh hien tai voi sigma = sigma_base * k^scale
      - Octave ke tiep: downsample 2x tu anh giua cua octave hien tai

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
            blurred = gaussian_blur_separable(current_img, sigma)
            octave_imgs.append(blurred)

        pyramid.append(octave_imgs)

        # Downsample: lay anh giua octave, giam kich thuoc 2x
        mid_img = octave_imgs[num_scales // 2]
        current_img = downsample_2x(mid_img)

    return pyramid


# =============================================================================
# PHAN 3: DoG (Difference of Gaussians)
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
# PHAN 4: LOAD ANH & SYNTHETIC IMAGE
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
    """Doc anh tu file → grayscale float32, range [0, 1]. Dung OpenCV de load."""
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
# PHAN 5: KIEM TRA TINH DUNG DAN (MAE so voi scipy)
# =============================================================================

def verify_correctness(image, num_octaves=4, num_scales=5, sigma_base=1.6,
                        mae_threshold=1e-3):
    """
    So sanh output cua phien ban no-lib voi scipy.ndimage.gaussian_filter.
    Tra ve (all_pass: bool, max_mae: float).
    """
    try:
        from scipy.ndimage import gaussian_filter as scipy_gaussian
    except ImportError:
        print("  [!] scipy khong co san — bo qua buoc verify.")
        return None

    pyramid_nolib = build_gaussian_pyramid_no_lib(
        image, num_octaves, num_scales, sigma_base)

    k = 2.0 ** (1.0 / num_scales)
    current_img = image.astype(np.float32)
    all_pass = True
    max_mae = 0.0

    for octave in range(num_octaves):
        for scale in range(num_scales):
            sigma = sigma_base * (k ** scale)
            ref = scipy_gaussian(current_img, sigma=sigma).astype(np.float32)
            our = pyramid_nolib[octave][scale]
            mae = float(np.mean(np.abs(our - ref)))
            max_mae = max(max_mae, mae)
            if mae >= mae_threshold:
                all_pass = False
                print(f"    [X] Octave {octave}, Scale {scale}: MAE={mae:.2e} >= {mae_threshold:.0e}")

        # Tinh anh dau octave ke tiep theo cach scipy de so sanh dung
        mid_idx = num_scales // 2
        sigma_mid = sigma_base * (k ** mid_idx)
        ref_mid = scipy_gaussian(current_img, sigma=sigma_mid).astype(np.float32)
        current_img = np.ascontiguousarray(ref_mid[::2, ::2])

    return all_pass, max_mae


# =============================================================================
# PHAN 6: MAIN
# =============================================================================

def main():
    print("=" * 65)
    print("  V0_1_no_lib.py -- Gaussian Pyramid (KHONG DUNG THU VIEN)")
    print("=" * 65)
    print("  Ky thuat: Separable Gaussian Blur (2x conv 1D) -- tu cai dat")
    print("  Khong dung: scipy, cv2.filter2D, hay ham convolution co san")
    print("=" * 65)

    NUM_OCTAVES = 4
    NUM_SCALES  = 5
    LIMIT_IMAGES = 10

    repo_root  = os.path.dirname(os.path.abspath(__file__))
    div2k_path = os.path.join(repo_root, "DIV2K_train_HR")

    image_paths = []
    if CV2_AVAILABLE and os.path.exists(div2k_path):
        image_paths = get_div2k_image_paths(div2k_path)

    # --- Kiem tra tinh dung dan tren anh nho ---
    print("\n[*] Kiem tra MAE (no-lib vs scipy) tren anh 256x256...")
    test_img = np.random.default_rng(0).random((256, 256)).astype(np.float32)
    result = verify_correctness(test_img, num_octaves=2, num_scales=3)
    if result is not None:
        passed, max_mae = result
        status = "[OK] PASS" if passed else "[!!] FAIL"
        print(f"  {status}  |  Max MAE toan pyramid = {max_mae:.2e}  (nguong 1e-3)")
    print()

    # --- Benchmark ---
    if image_paths:
        print(f"[DIR] Tim thay {len(image_paths)} anh trong: {div2k_path}")
        selected = image_paths[:LIMIT_IMAGES]
        print(f"[RUN] Benchmark {len(selected)} anh dau tien...\n")

        times = []
        for idx, path in enumerate(selected):
            fname = os.path.basename(path)
            img = load_single_image(path)
            H, W = img.shape

            t0 = time.perf_counter()
            pyramid = build_gaussian_pyramid_no_lib(img, NUM_OCTAVES, NUM_SCALES)
            dog = compute_dog_pyramid(pyramid)
            elapsed = time.perf_counter() - t0

            times.append(elapsed)
            print(f"  [{idx+1:2d}/{len(selected)}] {fname} ({H}x{W}): "
                  f"{elapsed*1000:.1f} ms")

        avg_time   = float(np.mean(times))
        total_time = float(np.sum(times))
        print(f"\n[RESULT] Ket qua {len(selected)} anh DIV2K (No-Lib):")
        print(f"  Tong thoi gian : {total_time:.3f} s")
        print(f"  Trung binh/anh : {avg_time*1000:.1f} ms")

        print("\n[PROFILE] cProfile breakdown (anh dau tien):")
        first_img = load_single_image(selected[0])
        pr = cProfile.Profile()
        pr.enable()
        build_gaussian_pyramid_no_lib(first_img, NUM_OCTAVES, NUM_SCALES)
        pr.disable()
        sio = io.StringIO()
        ps = pstats.Stats(pr, stream=sio).sort_stats("cumulative")
        ps.print_stats(10)
        print(sio.getvalue())

    else:
        print("[WARN] Khong tim thay anh DIV2K -- dung anh synthetic (2040x1356).")
        image = create_synthetic_div2k_image(seed=42)
        H, W = image.shape
        print(f"  Kich thuoc: {H}x{W} pixels\n")

        t0 = time.perf_counter()
        pyramid = build_gaussian_pyramid_no_lib(image, NUM_OCTAVES, NUM_SCALES)
        dog = compute_dog_pyramid(pyramid)
        elapsed = time.perf_counter() - t0
        print(f"  Thoi gian xu ly: {elapsed*1000:.1f} ms")

        print("\n[PROFILE] cProfile breakdown:")
        pr = cProfile.Profile()
        pr.enable()
        build_gaussian_pyramid_no_lib(image, NUM_OCTAVES, NUM_SCALES)
        pr.disable()
        sio = io.StringIO()
        ps = pstats.Stats(pr, stream=sio).sort_stats("cumulative")
        ps.print_stats(10)
        print(sio.getvalue())

        print("\n[*] Verify MAE (no-lib vs scipy) tren anh synthetic full-size...")
        result = verify_correctness(image, NUM_OCTAVES, NUM_SCALES)
        if result is not None:
            passed, max_mae = result
            status = "[OK] PASS" if passed else "[!!] FAIL"
            print(f"  {status}  |  Max MAE = {max_mae:.2e}  (nguong 1e-3)")

    print("\n[DONE] V0_1_no_lib.py hoan thanh.")


if __name__ == "__main__":
    main()
