from __future__ import annotations

import json
import math
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# =========================================================
# CONFIGURATION
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = PROJECT_ROOT / "images"

OUTPUT_ROOT = Path(__file__).resolve().parent / "dataset"
RGB_DIR = OUTPUT_ROOT / "rgb"
MASK_DIR = OUTPUT_ROOT / "masks"
PREVIEW_DIR = OUTPUT_ROOT / "previews"
TEMP_MASK_DIR = OUTPUT_ROOT / "_full_masks"

IMAGE_SIZE = 256
CAMERA_RADIUS = 3.0

# Approximate phone horizontal field of view.
# Used only to estimate focal length after the stable crop.
ORIGINAL_HFOV_DEG = 69.0

# If the object rotates clockwise as image index increases,
# the equivalent virtual camera orbit is the opposite direction.
ANGLE_SIGN = -1.0

RINGS = {
    "horizontal": {"step_deg": 10.0, "elevation_deg": 0.0},
    "upper": {"step_deg": 20.0, "elevation_deg": 25.0},
    "upper45": {"step_deg": 20.0, "elevation_deg": 45.0},
}

VALID_EXTS = {
    ".jpg", ".jpeg", ".png", ".heic",
    ".JPG", ".JPEG", ".PNG", ".HEIC"
}

# Background-removal model.
MODEL_ID = "ZhengPeng7/BiRefNet"
MODEL_INPUT_SIZE = 1024

# Try several thresholds and automatically choose a plausible centered mask.
MASK_THRESHOLDS = [0.50, 0.40, 0.30, 0.60, 0.70]

# A full-image foreground mask should normally occupy a modest fraction
# of the frame for this turntable capture.
MIN_FULL_FG_RATIO = 0.015
MAX_FULL_FG_RATIO = 0.55

# Stable crop padding around the detected owl.
CROP_PADDING = 1.22

# =========================================================
# HELPERS
# =========================================================

def natural_files(folder: Path):
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix in VALID_EXTS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def choose_device() -> torch.device:
    # BiRefNet segmentation is run on CPU in float32 for compatibility
    # with Apple Silicon / Python 3.14. NeRF training can still use MPS.
    return torch.device("cpu")


def load_model(device: torch.device):
    print(f"Loading BiRefNet segmentation on {device} (float32) ...")
    print("First run may download ~450 MB of model weights.")

    model = AutoModelForImageSegmentation.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
    )
    model.eval()
    model = model.float()
    model.to(device)
    return model


SEGMENT_TRANSFORM = transforms.Compose([
    transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])


def infer_soft_mask(model, device: torch.device, image_path: Path):
    """
    Returns a float32 mask in [0,1] at original image resolution.
    Falls back to CPU automatically if MPS inference fails.
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # (width, height)

    tensor = (
        SEGMENT_TRANSFORM(image)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
    )

    with torch.inference_mode():
        pred = model(tensor)[-1].sigmoid()

    pred = pred[0].squeeze().detach().float().cpu().numpy()

    pred = cv2.resize(
        pred,
        original_size,
        interpolation=cv2.INTER_LINEAR,
    )

    pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
    return pred, device


def central_component(binary: np.ndarray) -> np.ndarray:
    """
    Keep the most plausible central foreground structure.
    Internal holes remain holes.
    """
    h, w = binary.shape

    # Small cleanup. Avoid aggressive closing so carved holes remain visible.
    k3 = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k3, iterations=1)

    # A tiny close helps connect narrow adjacent parts without filling large holes.
    k5 = np.ones((5, 5), np.uint8)
    bridge = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, k5, iterations=1)

    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        bridge, 8
    )

    if n <= 1:
        return cleaned

    target_x = w * 0.50
    target_y = h * 0.62
    diag = math.sqrt(w * w + h * h)

    candidates = []

    for idx in range(1, n):
        x, y, ww, hh, area = stats[idx]
        cx, cy = centroids[idx]

        if area < max(250, int(h * w * 0.0003)):
            continue

        # Reject obvious border-sized background components.
        touches_many_borders = (
            (x <= 2) +
            (y <= 2) +
            (x + ww >= w - 2) +
            (y + hh >= h - 2)
        ) >= 2

        if touches_many_borders:
            continue

        distance = math.sqrt(
            (cx - target_x) ** 2 +
            (cy - target_y) ** 2
        ) / max(diag, 1.0)

        # Prefer large, central components.
        score = float(area) * math.exp(-8.0 * distance)
        candidates.append((score, idx, area))

    if not candidates:
        return cleaned

    candidates.sort(reverse=True)
    best_idx = candidates[0][1]
    best_area = candidates[0][2]

    result = np.zeros_like(cleaned)

    # Keep the main component plus meaningful nearby components.
    bx, by, bw, bh, _ = stats[best_idx]
    margin_x = int(bw * 0.30)
    margin_y = int(bh * 0.30)

    region_x0 = max(0, bx - margin_x)
    region_y0 = max(0, by - margin_y)
    region_x1 = min(w, bx + bw + margin_x)
    region_y1 = min(h, by + bh + margin_y)

    for idx in range(1, n):
        x, y, ww, hh, area = stats[idx]
        cx, cy = centroids[idx]

        in_region = (
            region_x0 <= cx <= region_x1 and
            region_y0 <= cy <= region_y1
        )

        if idx == best_idx or (in_region and area >= best_area * 0.006):
            # Use the ORIGINAL cleaned mask pixels for this label area,
            # rather than filling the whole connected region.
            region = labels == idx
            result[region & (cleaned > 0)] = 255

    # If the bridge caused some original object pixels to disappear from labels,
    # intersecting with a dilated result recovers nearby true foreground.
    dilated = cv2.dilate(result, k5, iterations=2)
    result = np.where((dilated > 0) & (cleaned > 0), 255, 0).astype(np.uint8)

    return result


def mask_from_soft(soft: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Try several thresholds and select the most plausible centered mask.
    Returns (binary_mask, selected_threshold).
    """
    h, w = soft.shape
    total = float(h * w)

    best = None

    for threshold in MASK_THRESHOLDS:
        binary = (soft >= threshold).astype(np.uint8) * 255
        mask = central_component(binary)

        ratio = float(np.mean(mask > 0))

        ys, xs = np.where(mask > 0)
        if len(xs) < 100:
            continue

        cx = float(xs.mean()) / w
        cy = float(ys.mean()) / h

        central_penalty = abs(cx - 0.50) + 0.7 * abs(cy - 0.62)

        plausible = (
            MIN_FULL_FG_RATIO <= ratio <= MAX_FULL_FG_RATIO
        )

        # Prefer plausible area and center location.
        score = (
            (2.0 if plausible else 0.0)
            - 2.0 * central_penalty
            - abs(ratio - 0.15)
        )

        item = (score, mask, threshold, ratio)

        if best is None or score > best[0]:
            best = item

    if best is None:
        # Last-resort threshold.
        threshold = 0.40
        mask = (soft >= threshold).astype(np.uint8) * 255
        mask = central_component(mask)
        return mask, threshold

    return best[1], best[2]


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)

    if len(xs) < 100:
        return None

    return (
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    )


def robust_ring_crop(
    bboxes,
    image_shape,
    padding=CROP_PADDING,
):
    """
    Create one stable square crop for every image in the same ring.
    Uses robust percentiles so one imperfect mask cannot destroy the crop.
    """
    h, w = image_shape[:2]

    valid = [b for b in bboxes if b is not None]

    if not valid:
        side = int(min(h, w) * 0.75)
        cx = w * 0.50
        cy = h * 0.62
    else:
        centers_x = []
        centers_y = []
        widths = []
        heights = []

        for x0, y0, x1, y1 in valid:
            centers_x.append((x0 + x1) / 2.0)
            centers_y.append((y0 + y1) / 2.0)
            widths.append(x1 - x0 + 1)
            heights.append(y1 - y0 + 1)

        cx = float(np.median(centers_x))
        cy = float(np.median(centers_y))

        # High percentile captures the object across rotations while ignoring
        # one catastrophic segmentation outlier.
        width_needed = float(np.percentile(widths, 95))
        height_needed = float(np.percentile(heights, 95))

        side = int(max(width_needed, height_needed) * padding)
        side = max(side, int(min(h, w) * 0.25))
        side = min(side, min(h, w))

    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))

    x0 = max(0, min(x0, w - side))
    y0 = max(0, min(y0, h - side))

    return x0, y0, side


def look_at_pose(
    theta_deg: float,
    elevation_deg: float,
    radius: float,
) -> np.ndarray:

    theta = math.radians(theta_deg)
    phi = math.radians(elevation_deg)

    pos = np.array([
        radius * math.sin(theta) * math.cos(phi),
        radius * math.sin(phi),
        radius * math.cos(theta) * math.cos(phi),
    ], dtype=np.float32)

    target = np.zeros(3, dtype=np.float32)

    forward = target - pos
    forward /= np.linalg.norm(forward) + 1e-8

    world_up = np.array(
        [0.0, 1.0, 0.0],
        dtype=np.float32,
    )

    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right) + 1e-8

    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-8

    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -forward
    c2w[:3, 3] = pos

    return c2w


def make_preview(
    rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Side-by-side preview: RGB | mask.
    """
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    label_rgb = rgb.copy()
    label_mask = mask_bgr.copy()

    cv2.putText(
        label_rgb,
        "RGB",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    cv2.putText(
        label_mask,
        "MASK",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    return np.hstack([label_rgb, label_mask])


# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 64)
    print(" ML DATASET PREPARATION - BiRefNet VERSION")
    print("=" * 64)
    print("Project root:", PROJECT_ROOT)

    # Remove stale generated data from the failed masks.
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)

    RGB_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_MASK_DIR.mkdir(parents=True, exist_ok=True)

    device = choose_device()

    try:
        model = load_model(device)
    except Exception as exc:
        print("\nERROR: Could not load BiRefNet.")
        print(exc)
        print("\nMake sure the extra packages are installed:")
        print(
            'pip install "transformers>=5.0.0" '
            'timm kornia einops accelerate huggingface_hub safetensors'
        )
        return

    frames = []

    for ring, cfg in RINGS.items():
        folder = IMAGE_ROOT / ring

        if not folder.exists():
            print(f"\n{ring}: folder not found -> {folder}")
            continue

        files = natural_files(folder)

        if not files:
            print(f"\n{ring}: no images found")
            continue

        first_img = cv2.imread(str(files[0]))

        if first_img is None:
            print(f"{ring}: first image cannot be read")
            continue

        orig_h, orig_w = first_img.shape[:2]

        print("\n" + "-" * 64)
        print(f"{ring.upper()}: {len(files)} images")
        print("Pass 1/2: neural foreground segmentation")

        bboxes = []
        thresholds = []
        full_ratios = []

        ring_temp = TEMP_MASK_DIR / ring
        ring_temp.mkdir(parents=True, exist_ok=True)

        # -------------------------------------------------
        # PASS 1: segment all original images and collect bbox
        # -------------------------------------------------

        for idx, path in enumerate(files):
            print(f"  [{idx + 1:02d}/{len(files):02d}] {path.name}", end="")

            try:
                soft, actual_device = infer_soft_mask(
                    model,
                    device,
                    path,
                )

                # If inference permanently fell back to CPU, use CPU from now on.
                if actual_device.type != device.type:
                    device = actual_device

                mask, threshold = mask_from_soft(soft)
                ratio = float(np.mean(mask > 0))
                bbox = bbox_from_mask(mask)

                temp_path = ring_temp / f"{idx:03d}.png"
                cv2.imwrite(str(temp_path), mask)

                bboxes.append(bbox)
                thresholds.append(threshold)
                full_ratios.append(ratio)

                status = "OK"
                if ratio < MIN_FULL_FG_RATIO or ratio > MAX_FULL_FG_RATIO:
                    status = "CHECK"

                print(
                    f" -> fg={ratio*100:5.1f}% "
                    f"thr={threshold:.2f} {status}"
                )

            except Exception as exc:
                print(" -> FAILED:", exc)
                bboxes.append(None)
                thresholds.append(None)
                full_ratios.append(0.0)

        # -------------------------------------------------
        # Stable crop for the whole ring
        # -------------------------------------------------

        crop_x, crop_y, crop_side = robust_ring_crop(
            bboxes,
            first_img.shape,
        )

        fx_orig = (
            0.5 * orig_w /
            math.tan(math.radians(ORIGINAL_HFOV_DEG) / 2.0)
        )

        fx = fx_orig * (IMAGE_SIZE / crop_side)
        fy = fx
        cx = IMAGE_SIZE / 2.0
        cy = IMAGE_SIZE / 2.0

        print("\nStable ring crop:")
        print(
            f"  x={crop_x}, y={crop_y}, side={crop_side}"
        )
        print(
            f"  estimated focal length after crop: {fx:.1f}px"
        )

        print("Pass 2/2: crop, resize, composite, save")

        # -------------------------------------------------
        # PASS 2: use SAME crop for every frame in ring
        # -------------------------------------------------

        preview_images = []

        for idx, path in enumerate(files):
            img = cv2.imread(str(path))
            mask_full = cv2.imread(
                str(ring_temp / f"{idx:03d}.png"),
                cv2.IMREAD_GRAYSCALE,
            )

            if img is None or mask_full is None:
                print("  skipped:", path.name)
                continue

            crop = img[
                crop_y:crop_y + crop_side,
                crop_x:crop_x + crop_side,
            ]

            mask_crop = mask_full[
                crop_y:crop_y + crop_side,
                crop_x:crop_x + crop_side,
            ]

            if (
                crop.shape[0] != crop_side or
                crop.shape[1] != crop_side
            ):
                print("  skipped bad crop:", path.name)
                continue

            crop_resized = cv2.resize(
                crop,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_AREA,
            )

            mask_resized = cv2.resize(
                mask_crop,
                (IMAGE_SIZE, IMAGE_SIZE),
                interpolation=cv2.INTER_NEAREST,
            )

            # Binary cleanup AFTER resize.
            mask_resized = np.where(
                mask_resized > 127,
                255,
                0,
            ).astype(np.uint8)

            # White background composite.
            alpha = (
                mask_resized.astype(np.float32) / 255.0
            )[..., None]

            rgb = (
                crop_resized.astype(np.float32) * alpha +
                255.0 * (1.0 - alpha)
            )

            rgb = np.clip(
                rgb,
                0,
                255,
            ).astype(np.uint8)

            stem = f"{ring}_{idx:03d}"

            rgb_path = RGB_DIR / f"{stem}.png"
            mask_path = MASK_DIR / f"{stem}.png"
            preview_path = PREVIEW_DIR / f"{stem}.jpg"

            cv2.imwrite(str(rgb_path), rgb)
            cv2.imwrite(str(mask_path), mask_resized)

            preview = make_preview(rgb, mask_resized)
            cv2.imwrite(str(preview_path), preview)

            # Foreground percentage inside the final 256x256 training crop.
            final_ratio = float(
                np.mean(mask_resized > 0)
            )

            if final_ratio < 0.08 or final_ratio > 0.90:
                print(
                    f"  WARNING {stem}: final foreground "
                    f"{final_ratio*100:.1f}%"
                )

            theta = (
                ANGLE_SIGN *
                idx *
                cfg["step_deg"]
            )

            pose = look_at_pose(
                theta,
                cfg["elevation_deg"],
                CAMERA_RADIUS,
            )

            frames.append({
                "ring": ring,
                "index": idx,
                "source_name": path.name,
                "theta_deg": theta,
                "elevation_deg": cfg["elevation_deg"],
                "rgb": str(
                    rgb_path.relative_to(OUTPUT_ROOT)
                ),
                "mask": str(
                    mask_path.relative_to(OUTPUT_ROOT)
                ),
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
                "transform_matrix": pose.tolist(),
                "mask_threshold": thresholds[idx],
                "full_image_foreground_ratio": full_ratios[idx],
                "final_crop_foreground_ratio": final_ratio,
            })

        print(f"{ring.upper()} complete.")

    metadata = {
        "image_size": IMAGE_SIZE,
        "camera_radius": CAMERA_RADIUS,
        "white_background": True,
        "angle_sign": ANGLE_SIGN,
        "segmentation_model": MODEL_ID,
        "frames": frames,
    }

    with open(
        OUTPUT_ROOT / "dataset.json",
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    # Delete temporary full-resolution masks after successful processing.
    if TEMP_MASK_DIR.exists():
        shutil.rmtree(TEMP_MASK_DIR)

    print("\n" + "=" * 64)
    print("DATASET READY")
    print("=" * 64)
    print("Frames:", len(frames))
    print("RGB:", RGB_DIR)
    print("Masks:", MASK_DIR)
    print("Previews:", PREVIEW_DIR)
    print("Metadata:", OUTPUT_ROOT / "dataset.json")
    print()
    print("NEXT:")
    print("1. Open dataset/previews")
    print("2. Check horizontal, upper, and upper45")
    print("3. Do NOT train until the owl is fully white in the masks")


if __name__ == "__main__":
    main()
