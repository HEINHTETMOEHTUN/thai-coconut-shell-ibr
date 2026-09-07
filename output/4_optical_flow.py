import cv2
import numpy as np
import glob
import os
import shutil


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

RINGS = [
    "horizontal",
    "upper"
]

PREPARED_ROOT = "prepared"

FLOW_ROOT = "optical_flows"

FLOW_VIS_ROOT = (
    "flow_visualizations"
)


# -------------------------------------------------
# FARNEBACK OPTICAL FLOW
# -------------------------------------------------

def compute_farneback(
    gray1,
    gray2
):

    flow = cv2.calcOpticalFlowFarneback(

        gray1,
        gray2,

        None,

        # Pyramid scale
        0.5,

        # Pyramid levels
        5,

        # Window size
        25,

        # Iterations
        5,

        # Polynomial neighborhood
        7,

        # Polynomial sigma
        1.5,

        0
    )

    return flow


# -------------------------------------------------
# FLOW VISUALIZATION
# -------------------------------------------------

def flow_to_bgr(flow):

    fx = flow[:, :, 0]
    fy = flow[:, :, 1]

    magnitude, angle = (
        cv2.cartToPolar(
            fx,
            fy
        )
    )

    hsv = np.zeros(
        (
            flow.shape[0],
            flow.shape[1],
            3
        ),
        dtype=np.uint8
    )

    hsv[:, :, 0] = (
        angle *
        180 /
        np.pi /
        2
    )

    hsv[:, :, 1] = 255

    hsv[:, :, 2] = (
        cv2.normalize(
            magnitude,
            None,
            0,
            255,
            cv2.NORM_MINMAX
        ).astype(
            np.uint8
        )
    )

    bgr = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    return bgr


# -------------------------------------------------
# PROCESS RINGS
# -------------------------------------------------

for ring in RINGS:

    images = sorted(
        glob.glob(
            os.path.join(
                PREPARED_ROOT,
                ring,
                "*.JPG"
            )
        )
    )

    if len(images) < 2:

        print(
            f"Not enough images "
            f"in {ring}"
        )

        continue

    flow_folder = os.path.join(
        FLOW_ROOT,
        ring
    )

    vis_folder = os.path.join(
        FLOW_VIS_ROOT,
        ring
    )

    # Delete old outputs
    if os.path.exists(
        flow_folder
    ):

        shutil.rmtree(
            flow_folder
        )

    if os.path.exists(
        vis_folder
    ):

        shutil.rmtree(
            vis_folder
        )

    os.makedirs(
        flow_folder,
        exist_ok=True
    )

    os.makedirs(
        vis_folder,
        exist_ok=True
    )

    print(
        f"\nProcessing "
        f"bidirectional optical flow: "
        f"{ring}"
    )

    print(
        "Images found:",
        len(images)
    )

    # -------------------------------------------------
    # EACH IMAGE PAIR
    # -------------------------------------------------

    for i in range(
        len(images) - 1
    ):

        img_a = cv2.imread(
            images[i]
        )

        img_b = cv2.imread(
            images[i + 1]
        )

        if (
            img_a is None
            or
            img_b is None
        ):

            print(
                f"Could not read "
                f"pair {i}"
            )

            continue

        img_b = cv2.resize(
            img_b,
            (
                img_a.shape[1],
                img_a.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )

        gray_a = cv2.cvtColor(
            img_a,
            cv2.COLOR_BGR2GRAY
        )

        gray_b = cv2.cvtColor(
            img_b,
            cv2.COLOR_BGR2GRAY
        )

        # Gentle Gaussian smoothing
        # reduces noise and tiny
        # lighting differences

        gray_a = cv2.GaussianBlur(
            gray_a,
            (5, 5),
            0
        )

        gray_b = cv2.GaussianBlur(
            gray_b,
            (5, 5),
            0
        )

        # ---------------------------------------------
        # FORWARD FLOW
        # A -> B
        # ---------------------------------------------

        flow_fwd = compute_farneback(
            gray_a,
            gray_b
        )

        # ---------------------------------------------
        # BACKWARD FLOW
        # B -> A
        # ---------------------------------------------

        flow_bwd = compute_farneback(
            gray_b,
            gray_a
        )

        # ---------------------------------------------
        # SAVE FLOWS
        # ---------------------------------------------

        np.save(

            os.path.join(
                flow_folder,
                f"flow_fwd_{i:02d}.npy"
            ),

            flow_fwd
        )

        np.save(

            os.path.join(
                flow_folder,
                f"flow_bwd_{i:02d}.npy"
            ),

            flow_bwd
        )

        # ---------------------------------------------
        # SAVE VISUALIZATIONS
        # ---------------------------------------------

        cv2.imwrite(

            os.path.join(
                vis_folder,
                f"flow_fwd_{i:02d}.jpg"
            ),

            flow_to_bgr(
                flow_fwd
            )
        )

        cv2.imwrite(

            os.path.join(
                vis_folder,
                f"flow_bwd_{i:02d}.jpg"
            ),

            flow_to_bgr(
                flow_bwd
            )
        )

        print(
            f"{ring}: "
            f"view {i+1} "
            f"<-> "
            f"view {i+2}"
        )


print(
    "\nAll bidirectional "
    "optical flows finished!"
)