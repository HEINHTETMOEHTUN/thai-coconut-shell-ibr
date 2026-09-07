import cv2
import numpy as np
import os
import glob
import re
import csv
import time
import math
import shutil

# =========================================================
# RESEARCH EVALUATION — NOVEL-VIEW RENDERING
# =========================================================
# Hold-out test:
#   View i-1 + View i+1  -> predict View i
#   Real View i is NEVER used to generate its prediction.
#
# Four methods are evaluated:
#   A. Baseline: one-way flow + simple linear blend
#   B. + Lighting normalization
#   C. + Bidirectional flow + consistency/confidence
#   D. Full proposed: C + Laplacian pyramid blending
#
# Metrics (object-focused when automatic mask succeeds):
#   MAE  ↓ lower is better
#   PSNR ↑ higher is better
#   SSIM ↑ higher is better
#   Edge F1 ↑ higher is better
#   Runtime ↓ lower is faster
# =========================================================

RINGS = ["horizontal", "upper"]
TARGET_SIZE = (800, 800)

# Test every other internal view so each target is exactly between two endpoints:
# 01 + 03 -> 02, 03 + 05 -> 04, ...
TEST_STRIDE = 2

# Set to an integer (e.g. 10) to cap cases per ring, or None to use all.
MAX_CASES_PER_RING = None

# Automatic foreground mask. If it fails, code falls back to full frame.
USE_OBJECT_MASK = True

# Lighting normalization settings (same idea as 1_prepare.py)
MIN_EXPOSURE_GAIN = 0.70
MAX_EXPOSURE_GAIN = 1.40

# Flow consistency settings (same as V2 pipeline)
BASE_THRESHOLD = 1.0
RELATIVE_THRESHOLD = 0.05
CONFIDENCE_SIGMA = 2.0
LOW_CONFIDENCE_SUM = 0.08
EPS = 1e-6
LAPLACIAN_LEVELS = 5

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
RAW_ROOT = os.path.join(PROJECT_ROOT, "images")
EVAL_ROOT = os.path.join(PROJECT_ROOT, "evaluation", "rendering")
COMPARISON_ROOT = os.path.join(EVAL_ROOT, "comparisons")
MASK_ROOT = os.path.join(EVAL_ROOT, "masks")
CHART_ROOT = os.path.join(EVAL_ROOT, "charts")

METHODS = [
    "A_Baseline",
    "B_Lighting",
    "C_Bidirectional_Occlusion",
    "D_Full_Proposed",
]


def natural_key(path):
    name = os.path.basename(path)
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", name)]


def list_images(folder):
    patterns = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG", "*.heic", "*.HEIC"]
    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(folder, pattern)))
    return sorted(set(files), key=natural_key)


def read_resize(path):
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)


# =========================================================
# LIGHTING NORMALIZATION
# =========================================================

def gray_world_white_balance(img):
    img_f = img.astype(np.float32)
    b_mean, g_mean, r_mean = cv2.mean(img)[:3]
    gray_mean = (b_mean + g_mean + r_mean) / 3.0
    gains = np.array([
        gray_mean / (b_mean + 1e-6),
        gray_mean / (g_mean + 1e-6),
        gray_mean / (r_mean + 1e-6),
    ], dtype=np.float32)
    img_f *= gains.reshape(1, 1, 3)
    return np.clip(img_f, 0, 255).astype(np.uint8)


def luminance_mean(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    return float(np.mean(lab[:, :, 0]))


def normalize_exposure(img, target_l_mean):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    current = float(np.mean(l))
    if current < 1e-6:
        return img
    gain = float(np.clip(target_l_mean / current, MIN_EXPOSURE_GAIN, MAX_EXPOSURE_GAIN))
    corrected_l = np.clip(l.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.merge([corrected_l, a, b]), cv2.COLOR_LAB2BGR)


def prepare_ring(raw_images):
    """Return raw-resized and normalized versions, preserving image order."""
    resized = []
    wb = []
    valid_paths = []

    for path in raw_images:
        img = read_resize(path)
        if img is None:
            print("Could not read:", path)
            continue
        resized.append(img)
        wb.append(gray_world_white_balance(img))
        valid_paths.append(path)

    if not wb:
        return [], [], []

    target_l = float(np.median([luminance_mean(im) for im in wb]))
    normalized = [normalize_exposure(im, target_l) for im in wb]
    return valid_paths, resized, normalized


# =========================================================
# OPTICAL FLOW
# =========================================================

def gray_for_flow(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def flow_old(img_a, img_b):
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)
    return cv2.calcOpticalFlowFarneback(
        gray_a, gray_b, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )


def flow_new(img_a, img_b):
    return cv2.calcOpticalFlowFarneback(
        gray_for_flow(img_a), gray_for_flow(img_b), None,
        0.5, 5, 25, 5, 7, 1.5, 0
    )


def make_grid(h, w):
    x, y = np.meshgrid(
        np.arange(w, dtype=np.float32),
        np.arange(h, dtype=np.float32),
    )
    return x, y


# =========================================================
# BASELINE METHOD
# =========================================================

def baseline_midpoint(img_a, img_b):
    """Replicates the old one-way-flow + simple blend idea at alpha=0.5."""
    flow = flow_old(img_a, img_b)
    h, w = img_a.shape[:2]
    x, y = make_grid(h, w)
    alpha = 0.5

    # Same mapping convention as the original baseline implementation.
    map_x1 = (x + flow[:, :, 0] * alpha).astype(np.float32)
    map_y1 = (y + flow[:, :, 1] * alpha).astype(np.float32)
    map_x2 = (x - flow[:, :, 0] * (1.0 - alpha)).astype(np.float32)
    map_y2 = (y - flow[:, :, 1] * (1.0 - alpha)).astype(np.float32)

    warp_a = cv2.remap(img_a, map_x1, map_y1, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    warp_b = cv2.remap(img_b, map_x2, map_y2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return cv2.addWeighted(warp_a, 0.5, warp_b, 0.5, 0)


# =========================================================
# BIDIRECTIONAL CONSISTENCY + CONFIDENCE
# =========================================================

def remap_flow(flow, map_x, map_y):
    fx = cv2.remap(flow[:, :, 0], map_x, map_y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    fy = cv2.remap(flow[:, :, 1], map_x, map_y, cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return np.dstack([fx, fy]).astype(np.float32)


def compute_consistency(primary_flow, opposite_flow):
    h, w = primary_flow.shape[:2]
    x, y = make_grid(h, w)
    map_x = x + primary_flow[:, :, 0]
    map_y = y + primary_flow[:, :, 1]

    valid = (
        (map_x >= 0) & (map_x <= w - 1) &
        (map_y >= 0) & (map_y <= h - 1)
    )

    opposite_sampled = remap_flow(opposite_flow, map_x, map_y)
    residual = primary_flow + opposite_sampled
    error = np.linalg.norm(residual, axis=2)

    mag_primary = np.linalg.norm(primary_flow, axis=2)
    mag_opposite = np.linalg.norm(opposite_sampled, axis=2)
    threshold = BASE_THRESHOLD + RELATIVE_THRESHOLD * (mag_primary + mag_opposite)
    occluded = (~valid) | (error > threshold)

    confidence = np.exp(-0.5 * (error / CONFIDENCE_SIGMA) ** 2).astype(np.float32)
    confidence[~valid] = 0.0
    confidence[occluded] = 0.0
    return confidence


def warp_image_and_validity(img, flow, fraction):
    h, w = img.shape[:2]
    x, y = make_grid(h, w)
    map_x = (x - fraction * flow[:, :, 0]).astype(np.float32)
    map_y = (y - fraction * flow[:, :, 1]).astype(np.float32)

    warped = cv2.remap(
        img, map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )

    ones = np.ones((h, w), dtype=np.float32)
    valid = cv2.remap(
        ones, map_x, map_y, cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )
    return warped, valid, map_x, map_y


def warp_scalar_map(value_map, map_x, map_y):
    return cv2.remap(
        value_map.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0
    )


# =========================================================
# LAPLACIAN PYRAMID BLENDING
# =========================================================

def laplacian_blend(img_a, img_b, mask_a, levels=5):
    a = img_a.astype(np.float32) / 255.0
    b = img_b.astype(np.float32) / 255.0
    mask = np.clip(mask_a.astype(np.float32), 0.0, 1.0)

    h, w = mask.shape
    max_possible = max(1, int(np.floor(np.log2(min(h, w)))) - 2)
    levels = max(1, min(levels, max_possible))

    gp_a, gp_b, gp_m = [a], [b], [mask]
    for _ in range(levels):
        gp_a.append(cv2.pyrDown(gp_a[-1]))
        gp_b.append(cv2.pyrDown(gp_b[-1]))
        gp_m.append(cv2.pyrDown(gp_m[-1]))

    lp_a, lp_b = [], []
    for i in range(levels):
        size_a = (gp_a[i].shape[1], gp_a[i].shape[0])
        size_b = (gp_b[i].shape[1], gp_b[i].shape[0])
        lp_a.append(gp_a[i] - cv2.pyrUp(gp_a[i + 1], dstsize=size_a))
        lp_b.append(gp_b[i] - cv2.pyrUp(gp_b[i + 1], dstsize=size_b))
    lp_a.append(gp_a[-1])
    lp_b.append(gp_b[-1])

    blended_levels = []
    for la, lb, gm in zip(lp_a, lp_b, gp_m):
        m3 = gm[:, :, None]
        blended_levels.append(la * m3 + lb * (1.0 - m3))

    result = blended_levels[-1]
    for i in range(levels - 1, -1, -1):
        size = (blended_levels[i].shape[1], blended_levels[i].shape[0])
        result = cv2.pyrUp(result, dstsize=size) + blended_levels[i]

    return np.clip(result * 255.0, 0, 255).astype(np.uint8)


def bidirectional_midpoint(img_a, img_b, use_laplacian):
    """V2 method at alpha=0.5. If use_laplacian=False, this is the ablation C stage."""
    alpha = 0.5
    flow_fwd = flow_new(img_a, img_b)
    flow_bwd = flow_new(img_b, img_a)

    conf_a = compute_consistency(flow_fwd, flow_bwd)
    conf_b = compute_consistency(flow_bwd, flow_fwd)

    warp_a, valid_a, map_ax, map_ay = warp_image_and_validity(img_a, flow_fwd, alpha)
    warp_b, valid_b, map_bx, map_by = warp_image_and_validity(img_b, flow_bwd, 1.0 - alpha)

    conf_a_warp = warp_scalar_map(conf_a, map_ax, map_ay) * valid_a
    conf_b_warp = warp_scalar_map(conf_b, map_bx, map_by) * valid_b

    weight_a = (1.0 - alpha) * conf_a_warp
    weight_b = alpha * conf_b_warp
    weight_sum = weight_a + weight_b

    mask_a = np.zeros_like(weight_sum, dtype=np.float32)
    reliable = weight_sum > EPS
    mask_a[reliable] = weight_a[reliable] / weight_sum[reliable]

    low_confidence = weight_sum < LOW_CONFIDENCE_SUM
    choose_a = weight_a >= weight_b
    both_zero = weight_sum <= EPS
    choose_a[both_zero] = alpha <= 0.5
    mask_a[low_confidence & choose_a] = 1.0
    mask_a[low_confidence & (~choose_a)] = 0.0

    only_a_valid = (valid_a > 0.5) & (valid_b <= 0.5)
    only_b_valid = (valid_b > 0.5) & (valid_a <= 0.5)
    mask_a[only_a_valid] = 1.0
    mask_a[only_b_valid] = 0.0

    if use_laplacian:
        result = laplacian_blend(warp_a, warp_b, mask_a, LAPLACIAN_LEVELS)
    else:
        m3 = mask_a[:, :, None]
        result = np.clip(
            warp_a.astype(np.float32) * m3 +
            warp_b.astype(np.float32) * (1.0 - m3),
            0, 255
        ).astype(np.uint8)

    # Hard protection in extremely unreliable areas, same logic as final V2.
    very_low = weight_sum < (LOW_CONFIDENCE_SUM * 0.5)
    result[very_low & choose_a] = warp_a[very_low & choose_a]
    result[very_low & (~choose_a)] = warp_b[very_low & (~choose_a)]
    return result


# =========================================================
# AUTOMATIC OBJECT MASK
# =========================================================

def make_object_mask(img):
    """
    Estimate foreground from the border color in Lab space.
    Works best when the object is centered against a relatively plain backdrop.
    Returns uint8 mask {0,255}; may return full-frame mask as a safe fallback.
    """
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = max(8, min(h, w) // 30)

    samples = np.concatenate([
        lab[:border, :, :].reshape(-1, 3),
        lab[-border:, :, :].reshape(-1, 3),
        lab[:, :border, :].reshape(-1, 3),
        lab[:, -border:, :].reshape(-1, 3),
    ], axis=0)
    bg = np.median(samples, axis=0)
    dist = np.linalg.norm(lab - bg.reshape(1, 1, 3), axis=2)
    dist_u8 = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, mask = cv2.threshold(dist_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num_labels > 1:
        # Prefer a sizeable component reasonably close to image center.
        cx0, cy0 = w / 2.0, h / 2.0
        best_label, best_score = None, -1.0
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            x = stats[label, cv2.CC_STAT_LEFT]
            y = stats[label, cv2.CC_STAT_TOP]
            ww = stats[label, cv2.CC_STAT_WIDTH]
            hh = stats[label, cv2.CC_STAT_HEIGHT]
            cx, cy = x + ww / 2.0, y + hh / 2.0
            center_penalty = math.hypot((cx - cx0) / w, (cy - cy0) / h)
            score = area * max(0.2, 1.0 - center_penalty)
            if score > best_score:
                best_score, best_label = score, label
        mask = np.where(labels == best_label, 255, 0).astype(np.uint8)

    # Slight dilation includes silhouette edge where ghosting is most visible.
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=1)

    ratio = float(np.mean(mask > 0))
    if ratio < 0.02 or ratio > 0.90:
        return np.full((h, w), 255, dtype=np.uint8), False
    return mask, True


# =========================================================
# METRICS
# =========================================================

def masked_values(arr, mask):
    return arr[mask > 0]


def metric_mae(pred, gt, mask):
    diff = np.abs(pred.astype(np.float32) - gt.astype(np.float32))
    m = np.repeat((mask > 0)[:, :, None], 3, axis=2)
    vals = diff[m]
    return float(np.mean(vals)) if vals.size else float("nan")


def metric_mse(pred, gt, mask):
    diff2 = (pred.astype(np.float32) - gt.astype(np.float32)) ** 2
    m = np.repeat((mask > 0)[:, :, None], 3, axis=2)
    vals = diff2[m]
    return float(np.mean(vals)) if vals.size else float("nan")


def metric_psnr(pred, gt, mask):
    mse = metric_mse(pred, gt, mask)
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * math.log10((255.0 ** 2) / mse))


def ssim_map_gray(img1, img2):
    """Standard local-window SSIM map on grayscale images."""
    x = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    y = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu1 = cv2.GaussianBlur(x, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(y, (11, 11), 1.5)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = cv2.GaussianBlur(x * x, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(y * y, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(x * y, (11, 11), 1.5) - mu12

    numerator = (2 * mu12 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    return numerator / (denominator + 1e-12)


def metric_ssim(pred, gt, mask):
    ssim_map = ssim_map_gray(pred, gt)
    vals = ssim_map[mask > 0]
    return float(np.mean(vals)) if vals.size else float("nan")


def metric_edge_f1(pred, gt, mask, tolerance_px=2):
    pred_gray = cv2.cvtColor(pred, cv2.COLOR_BGR2GRAY)
    gt_gray = cv2.cvtColor(gt, cv2.COLOR_BGR2GRAY)
    pred_edges = cv2.Canny(pred_gray, 80, 160) > 0
    gt_edges = cv2.Canny(gt_gray, 80, 160) > 0
    roi = mask > 0
    pred_edges &= roi
    gt_edges &= roi

    kernel = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), np.uint8)
    pred_dil = cv2.dilate(pred_edges.astype(np.uint8), kernel) > 0
    gt_dil = cv2.dilate(gt_edges.astype(np.uint8), kernel) > 0

    pred_count = int(np.sum(pred_edges))
    gt_count = int(np.sum(gt_edges))
    if pred_count == 0 and gt_count == 0:
        return 1.0
    if pred_count == 0 or gt_count == 0:
        return 0.0

    precision = float(np.sum(pred_edges & gt_dil)) / pred_count
    recall = float(np.sum(gt_edges & pred_dil)) / gt_count
    if precision + recall <= 1e-12:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def compute_metrics(pred, gt, mask):
    return {
        "MAE": metric_mae(pred, gt, mask),
        "PSNR": metric_psnr(pred, gt, mask),
        "SSIM": metric_ssim(pred, gt, mask),
        "Edge_F1": metric_edge_f1(pred, gt, mask),
    }


# =========================================================
# VISUAL COMPARISON
# =========================================================

def labeled_tile(img, label):
    tile = img.copy()
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 48), (0, 0, 0), -1)
    cv2.putText(tile, label, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return tile


def save_comparison(path, gt, outputs):
    tiles = [labeled_tile(gt, "Ground Truth")]
    for method, img in outputs.items():
        short = {
            "A_Baseline": "A Baseline",
            "B_Lighting": "B +Lighting",
            "C_Bidirectional_Occlusion": "C +BiFlow/Occlusion",
            "D_Full_Proposed": "D Full Proposed",
        }[method]
        tiles.append(labeled_tile(img, short))

    # Downsize only for the montage; metrics always use original 800x800 images.
    thumb_size = (320, 320)
    tiles = [cv2.resize(t, thumb_size, interpolation=cv2.INTER_AREA) for t in tiles]
    row1 = np.hstack(tiles[:3])
    blank = np.zeros_like(tiles[0])
    row2 = np.hstack(tiles[3:] + [blank] * max(0, 3 - len(tiles[3:])))
    canvas = np.vstack([row1, row2])
    cv2.imwrite(path, canvas)


# =========================================================
# CSV + CHARTS
# =========================================================

def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_summary(rows):
    summary = []
    for method in METHODS:
        subset = [r for r in rows if r["Method"] == method]
        if not subset:
            continue
        summary.append({
            "Method": method,
            "Cases": len(subset),
            "MAE_mean": float(np.mean([float(r["MAE"]) for r in subset])),
            "PSNR_mean": float(np.mean([float(r["PSNR"]) for r in subset])),
            "SSIM_mean": float(np.mean([float(r["SSIM"]) for r in subset])),
            "Edge_F1_mean": float(np.mean([float(r["Edge_F1"]) for r in subset])),
            "Runtime_s_mean": float(np.mean([float(r["Runtime_s"]) for r in subset])),
        })
    return summary


def try_make_charts(summary):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("\nmatplotlib not installed — CSV results were still created.")
        print("Optional charts: python3 -m pip install matplotlib")
        return

    labels = [r["Method"].replace("_", "\n") for r in summary]
    chart_specs = [
        ("MAE_mean", "MAE (lower is better)", "mae.png"),
        ("PSNR_mean", "PSNR dB (higher is better)", "psnr.png"),
        ("SSIM_mean", "SSIM (higher is better)", "ssim.png"),
        ("Edge_F1_mean", "Edge F1 (higher is better)", "edge_f1.png"),
        ("Runtime_s_mean", "Runtime per prediction, seconds", "runtime.png"),
    ]
    for key, title, filename in chart_specs:
        values = [r[key] for r in summary]
        plt.figure(figsize=(9, 5))
        plt.bar(labels, values)
        plt.title(title)
        plt.ylabel(title)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(CHART_ROOT, filename), dpi=180)
        plt.close()


# =========================================================
# MAIN EVALUATION
# =========================================================

def main():
    if os.path.exists(EVAL_ROOT):
        shutil.rmtree(EVAL_ROOT)
    os.makedirs(COMPARISON_ROOT, exist_ok=True)
    os.makedirs(MASK_ROOT, exist_ok=True)
    os.makedirs(CHART_ROOT, exist_ok=True)

    all_rows = []
    total_cases = 0

    print("=========================================================")
    print(" NOVEL-VIEW RENDERING RESEARCH EVALUATION")
    print("=========================================================")
    print("Project root:", PROJECT_ROOT)

    for ring in RINGS:
        folder = os.path.join(RAW_ROOT, ring)
        raw_paths = list_images(folder)
        if len(raw_paths) < 3:
            print(f"\n{ring}: not enough images ({len(raw_paths)})")
            continue

        valid_paths, raw_resized, normalized = prepare_ring(raw_paths)
        n = len(valid_paths)
        if n < 3:
            continue

        # Targets: indices 1,3,5,... so left/right are exactly one step away.
        target_indices = list(range(1, n - 1, TEST_STRIDE))
        if MAX_CASES_PER_RING is not None:
            target_indices = target_indices[:MAX_CASES_PER_RING]

        print(f"\n{ring.upper()}: {n} valid images, {len(target_indices)} hold-out cases")

        for case_num, target_idx in enumerate(target_indices, start=1):
            left_idx = target_idx - 1
            right_idx = target_idx + 1

            left_raw = raw_resized[left_idx]
            target_raw = raw_resized[target_idx]
            right_raw = raw_resized[right_idx]

            left_norm = normalized[left_idx]
            target_norm = normalized[target_idx]
            right_norm = normalized[right_idx]

            # Ground truth for full-pipeline evaluation is the normalized held-out target.
            # This avoids scoring illumination differences as geometric/rendering errors.
            gt = target_norm

            if USE_OBJECT_MASK:
                mask, mask_ok = make_object_mask(gt)
            else:
                mask = np.full(gt.shape[:2], 255, dtype=np.uint8)
                mask_ok = False

            mask_path = os.path.join(MASK_ROOT, f"{ring}_case_{case_num:02d}_mask.jpg")
            cv2.imwrite(mask_path, mask)

            outputs = {}
            runtimes = {}

            # A — Old baseline on raw resized images.
            t0 = time.perf_counter()
            outputs["A_Baseline"] = baseline_midpoint(left_raw, right_raw)
            runtimes["A_Baseline"] = time.perf_counter() - t0

            # B — Same baseline algorithm, now with lighting-normalized inputs.
            t0 = time.perf_counter()
            outputs["B_Lighting"] = baseline_midpoint(left_norm, right_norm)
            runtimes["B_Lighting"] = time.perf_counter() - t0

            # C — Lighting + bidirectional flow + consistency/occlusion/confidence, simple blend.
            t0 = time.perf_counter()
            outputs["C_Bidirectional_Occlusion"] = bidirectional_midpoint(
                left_norm, right_norm, use_laplacian=False
            )
            runtimes["C_Bidirectional_Occlusion"] = time.perf_counter() - t0

            # D — Full proposed pipeline including Laplacian pyramid blending.
            t0 = time.perf_counter()
            outputs["D_Full_Proposed"] = bidirectional_midpoint(
                left_norm, right_norm, use_laplacian=True
            )
            runtimes["D_Full_Proposed"] = time.perf_counter() - t0

            for method, pred in outputs.items():
                # IMPORTANT: every method is scored against the SAME held-out
                # normalized ground-truth image. This keeps MAE/PSNR/SSIM/Edge-F1
                # directly comparable across the ablation stages. The baseline
                # is allowed to suffer from illumination mismatch because lighting
                # normalization is one of the proposed pipeline improvements.
                method_mask = mask
                metrics = compute_metrics(pred, gt, method_mask)

                all_rows.append({
                    "Ring": ring,
                    "Case": case_num,
                    "Left_View": os.path.basename(valid_paths[left_idx]),
                    "Held_Out_Target": os.path.basename(valid_paths[target_idx]),
                    "Right_View": os.path.basename(valid_paths[right_idx]),
                    "Method": method,
                    "Mask_OK": int(mask_ok),
                    "MAE": f"{metrics['MAE']:.6f}",
                    "PSNR": f"{metrics['PSNR']:.6f}",
                    "SSIM": f"{metrics['SSIM']:.6f}",
                    "Edge_F1": f"{metrics['Edge_F1']:.6f}",
                    "Runtime_s": f"{runtimes[method]:.6f}",
                })

            comparison_path = os.path.join(
                COMPARISON_ROOT, f"{ring}_case_{case_num:02d}.jpg"
            )
            save_comparison(comparison_path, gt, outputs)

            total_cases += 1
            print(
                f"  case {case_num:02d}: "
                f"{os.path.basename(valid_paths[left_idx])} + "
                f"{os.path.basename(valid_paths[right_idx])} -> "
                f"hold out {os.path.basename(valid_paths[target_idx])}"
            )

    if not all_rows:
        print("\nNo evaluation cases were created. Check images/horizontal and images/upper.")
        return

    per_case_path = os.path.join(EVAL_ROOT, "per_case_results.csv")
    write_csv(
        per_case_path,
        all_rows,
        [
            "Ring", "Case", "Left_View", "Held_Out_Target", "Right_View",
            "Method", "Mask_OK", "MAE", "PSNR", "SSIM", "Edge_F1", "Runtime_s"
        ],
    )

    summary = make_summary(all_rows)
    summary_path = os.path.join(EVAL_ROOT, "summary_results.csv")
    write_csv(
        summary_path,
        summary,
        ["Method", "Cases", "MAE_mean", "PSNR_mean", "SSIM_mean", "Edge_F1_mean", "Runtime_s_mean"],
    )

    try_make_charts(summary)

    print("\n=========================================================")
    print(" EVALUATION COMPLETE")
    print("=========================================================")
    print(f"Hold-out cases: {total_cases}")
    print("\nSummary:")
    print(f"{'Method':34s} {'MAE↓':>9s} {'PSNR↑':>9s} {'SSIM↑':>9s} {'EdgeF1↑':>9s} {'Time(s)':>9s}")
    for r in summary:
        print(
            f"{r['Method']:34s} "
            f"{r['MAE_mean']:9.3f} "
            f"{r['PSNR_mean']:9.3f} "
            f"{r['SSIM_mean']:9.4f} "
            f"{r['Edge_F1_mean']:9.4f} "
            f"{r['Runtime_s_mean']:9.3f}"
        )

    print("\nCreated:")
    print(per_case_path)
    print(summary_path)
    print(COMPARISON_ROOT)
    print(CHART_ROOT)
    print("\nIMPORTANT: report the actual measured values. Do not replace them with example numbers.")


if __name__ == "__main__":
    main()
