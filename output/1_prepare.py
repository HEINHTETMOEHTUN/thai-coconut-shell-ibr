import cv2
import os
import glob
import shutil
import numpy as np

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

RINGS = ["horizontal", "upper"]

INPUT_ROOT = "../images"
OUTPUT_ROOT = "prepared"

TARGET_SIZE = (800, 800)

# Prevent extreme brightness correction
MIN_EXPOSURE_GAIN = 0.70
MAX_EXPOSURE_GAIN = 1.40


# -------------------------------------------------
# GLOBAL WHITE BALANCE
# -------------------------------------------------

def gray_world_white_balance(img):

    img_f = img.astype(np.float32)

    b_mean, g_mean, r_mean = cv2.mean(img)[:3]

    gray_mean = (
        b_mean +
        g_mean +
        r_mean
    ) / 3.0

    eps = 1e-6

    gains = np.array(
        [
            gray_mean / (b_mean + eps),
            gray_mean / (g_mean + eps),
            gray_mean / (r_mean + eps)
        ],
        dtype=np.float32
    )

    img_f *= gains.reshape(1, 1, 3)

    img_f = np.clip(
        img_f,
        0,
        255
    )

    return img_f.astype(np.uint8)


# -------------------------------------------------
# GET IMAGE BRIGHTNESS
# -------------------------------------------------

def luminance_mean(img):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    lightness = lab[:, :, 0]

    return float(
        np.mean(lightness)
    )


# -------------------------------------------------
# GLOBAL EXPOSURE NORMALIZATION
# -------------------------------------------------

def normalize_exposure(
    img,
    target_l_mean
):

    lab = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    current_l_mean = float(
        np.mean(l)
    )

    if current_l_mean < 1e-6:
        return img

    gain = (
        target_l_mean /
        current_l_mean
    )

    gain = float(
        np.clip(
            gain,
            MIN_EXPOSURE_GAIN,
            MAX_EXPOSURE_GAIN
        )
    )

    corrected_l = np.clip(
        l.astype(np.float32) * gain,
        0,
        255
    ).astype(np.uint8)

    corrected_lab = cv2.merge(
        [
            corrected_l,
            a,
            b
        ]
    )

    corrected = cv2.cvtColor(
        corrected_lab,
        cv2.COLOR_LAB2BGR
    )

    return corrected


# -------------------------------------------------
# PROCESS EACH RING
# -------------------------------------------------

for ring in RINGS:

    input_folder = os.path.join(
        INPUT_ROOT,
        ring
    )

    output_folder = os.path.join(
        OUTPUT_ROOT,
        ring
    )

    # Delete old prepared folder
    if os.path.exists(output_folder):

        shutil.rmtree(
            output_folder
        )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    files = sorted(
        glob.glob(
            os.path.join(
                input_folder,
                "*"
            )
        )
    )

    print(
        f"\n{ring}: "
        f"{len(files)} files found"
    )

    # -------------------------------------------------
    # FIRST PASS
    # Resize + white balance
    # -------------------------------------------------

    temporary_images = []

    for filename in files:

        img = cv2.imread(
            filename
        )

        if img is None:

            print(
                "Could not read:",
                filename
            )

            continue

        img = cv2.resize(
            img,
            TARGET_SIZE,
            interpolation=cv2.INTER_AREA
        )

        # White balance
        img = gray_world_white_balance(
            img
        )

        temporary_images.append(
            (
                filename,
                img
            )
        )

    if len(temporary_images) == 0:

        print(
            f"No valid images in {ring}"
        )

        continue

    # -------------------------------------------------
    # Find median brightness of ring
    # -------------------------------------------------

    brightness_values = []

    for _, img in temporary_images:

        brightness_values.append(
            luminance_mean(img)
        )

    target_l_mean = float(
        np.median(
            brightness_values
        )
    )

    print(
        f"{ring}: "
        f"target luminance = "
        f"{target_l_mean:.2f}"
    )

    # -------------------------------------------------
    # SECOND PASS
    # Exposure normalization + save
    # -------------------------------------------------

    for output_index, (
        filename,
        img
    ) in enumerate(
        temporary_images,
        start=1
    ):

        img = normalize_exposure(
            img,
            target_l_mean
        )

        output_path = os.path.join(
            output_folder,
            f"view{output_index:02d}.JPG"
        )

        cv2.imwrite(
            output_path,
            img
        )

        print(
            "Prepared:",
            output_path
        )


print(
    "\nAll images prepared "
    "with lighting normalization!"
)