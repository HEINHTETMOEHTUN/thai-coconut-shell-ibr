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

FLOW_ROOT = "optical_flows"

OUTPUT_ROOT = (
    "flow_confidence"
)

# Larger values = more tolerant

BASE_THRESHOLD = 1.0

RELATIVE_THRESHOLD = 0.05

CONFIDENCE_SIGMA = 2.0


# -------------------------------------------------
# SAMPLE FLOW AT FLOAT COORDINATES
# -------------------------------------------------

def remap_flow(
    flow,
    map_x,
    map_y
):

    fx = cv2.remap(

        flow[:, :, 0],

        map_x,
        map_y,

        cv2.INTER_LINEAR,

        borderMode=(
            cv2.BORDER_CONSTANT
        ),

        borderValue=0
    )

    fy = cv2.remap(

        flow[:, :, 1],

        map_x,
        map_y,

        cv2.INTER_LINEAR,

        borderMode=(
            cv2.BORDER_CONSTANT
        ),

        borderValue=0
    )

    sampled = np.dstack(
        [
            fx,
            fy
        ]
    )

    return sampled.astype(
        np.float32
    )


# -------------------------------------------------
# FORWARD-BACKWARD CONSISTENCY
# -------------------------------------------------

def compute_consistency(
    primary_flow,
    opposite_flow
):

    h, w = (
        primary_flow.shape[:2]
    )

    x, y = np.meshgrid(

        np.arange(
            w,
            dtype=np.float32
        ),

        np.arange(
            h,
            dtype=np.float32
        )
    )

    # ---------------------------------------------
    # Follow primary flow
    #
    # p -> p + F(p)
    # ---------------------------------------------

    map_x = (
        x +
        primary_flow[:, :, 0]
    )

    map_y = (
        y +
        primary_flow[:, :, 1]
    )

    # ---------------------------------------------
    # Check whether the destination
    # remains inside image
    # ---------------------------------------------

    valid = (

        (map_x >= 0)

        &

        (map_x <= w - 1)

        &

        (map_y >= 0)

        &

        (map_y <= h - 1)

    )

    # ---------------------------------------------
    # Sample opposite flow
    # at mapped position
    # ---------------------------------------------

    opposite_sampled = (
        remap_flow(
            opposite_flow,
            map_x,
            map_y
        )
    )

    # ---------------------------------------------
    # FORWARD + BACKWARD
    #
    # Should approximately equal zero
    # ---------------------------------------------

    residual = (
        primary_flow +
        opposite_sampled
    )

    error = np.linalg.norm(
        residual,
        axis=2
    )

    # ---------------------------------------------
    # Motion magnitudes
    # ---------------------------------------------

    mag_primary = np.linalg.norm(
        primary_flow,
        axis=2
    )

    mag_opposite = np.linalg.norm(
        opposite_sampled,
        axis=2
    )

    # ---------------------------------------------
    # Adaptive consistency threshold
    # ---------------------------------------------

    threshold = (

        BASE_THRESHOLD

        +

        RELATIVE_THRESHOLD
        *
        (
            mag_primary +
            mag_opposite
        )
    )

    # ---------------------------------------------
    # OCCLUSION / BAD MATCH
    # ---------------------------------------------

    occluded = (

        (~valid)

        |

        (error > threshold)
    )

    # ---------------------------------------------
    # CONFIDENCE
    #
    # Small error = high confidence
    # Large error = low confidence
    # ---------------------------------------------

    confidence = np.exp(

        -0.5 *

        (
            error /
            CONFIDENCE_SIGMA
        ) ** 2

    ).astype(
        np.float32
    )

    # Invalid / occluded pixels
    # should not influence blending

    confidence[
        ~valid
    ] = 0.0

    confidence[
        occluded
    ] = 0.0

    return (
        error,
        confidence,
        occluded,
        valid
    )


# -------------------------------------------------
# PROCESS
# -------------------------------------------------

for ring in RINGS:

    forward_files = sorted(

        glob.glob(

            os.path.join(
                FLOW_ROOT,
                ring,
                "flow_fwd_*.npy"
            )

        )

    )

    backward_files = sorted(

        glob.glob(

            os.path.join(
                FLOW_ROOT,
                ring,
                "flow_bwd_*.npy"
            )

        )

    )

    if (

        len(forward_files) == 0

        or

        len(forward_files)
        !=
        len(backward_files)

    ):

        print(
            f"Flow files missing "
            f"or mismatched in {ring}"
        )

        continue

    output_folder = os.path.join(
        OUTPUT_ROOT,
        ring
    )

    if os.path.exists(
        output_folder
    ):

        shutil.rmtree(
            output_folder
        )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    print(
        f"\nChecking "
        f"forward-backward consistency: "
        f"{ring}"
    )

    # -------------------------------------------------
    # EACH PAIR
    # -------------------------------------------------

    for i, (
        forward_path,
        backward_path
    ) in enumerate(

        zip(
            forward_files,
            backward_files
        )

    ):

        flow_fwd = np.load(
            forward_path
        ).astype(
            np.float32
        )

        flow_bwd = np.load(
            backward_path
        ).astype(
            np.float32
        )

        # ---------------------------------------------
        # CONFIDENCE FROM A
        # ---------------------------------------------

        (
            error_a,
            conf_a,
            occ_a,
            valid_a

        ) = compute_consistency(

            flow_fwd,
            flow_bwd

        )

        # ---------------------------------------------
        # CONFIDENCE FROM B
        # ---------------------------------------------

        (
            error_b,
            conf_b,
            occ_b,
            valid_b

        ) = compute_consistency(

            flow_bwd,
            flow_fwd

        )

        # ---------------------------------------------
        # SAVE NUMERICAL FILES
        # ---------------------------------------------

        np.save(

            os.path.join(
                output_folder,
                f"conf_a_{i:02d}.npy"
            ),

            conf_a
        )

        np.save(

            os.path.join(
                output_folder,
                f"conf_b_{i:02d}.npy"
            ),

            conf_b
        )

        np.save(

            os.path.join(
                output_folder,
                f"occ_a_{i:02d}.npy"
            ),

            occ_a
        )

        np.save(

            os.path.join(
                output_folder,
                f"occ_b_{i:02d}.npy"
            ),

            occ_b
        )

        np.save(

            os.path.join(
                output_folder,
                f"error_a_{i:02d}.npy"
            ),

            error_a
        )

        np.save(

            os.path.join(
                output_folder,
                f"error_b_{i:02d}.npy"
            ),

            error_b
        )

        # ---------------------------------------------
        # SAVE VISUAL CONFIDENCE MAPS
        # ---------------------------------------------

        confidence_a_img = np.clip(
            conf_a * 255,
            0,
            255
        ).astype(
            np.uint8
        )

        confidence_b_img = np.clip(
            conf_b * 255,
            0,
            255
        ).astype(
            np.uint8
        )

        cv2.imwrite(

            os.path.join(
                output_folder,
                f"confidence_a_{i:02d}.jpg"
            ),

            confidence_a_img
        )

        cv2.imwrite(

            os.path.join(
                output_folder,
                f"confidence_b_{i:02d}.jpg"
            ),

            confidence_b_img
        )

        # ---------------------------------------------
        # SAVE OCCLUSION MASKS
        # ---------------------------------------------

        occlusion_a_img = (
            occ_a.astype(
                np.uint8
            )
            * 255
        )

        occlusion_b_img = (
            occ_b.astype(
                np.uint8
            )
            * 255
        )

        cv2.imwrite(

            os.path.join(
                output_folder,
                f"occlusion_a_{i:02d}.jpg"
            ),

            occlusion_a_img
        )

        cv2.imwrite(

            os.path.join(
                output_folder,
                f"occlusion_b_{i:02d}.jpg"
            ),

            occlusion_b_img
        )

        good_a = (
            100.0 *
            float(
                np.mean(
                    conf_a > 0
                )
            )
        )

        good_b = (
            100.0 *
            float(
                np.mean(
                    conf_b > 0
                )
            )
        )

        print(

            f"{ring} pair {i:02d}: "

            f"A-confidence pixels="
            f"{good_a:.1f}%  "

            f"B-confidence pixels="
            f"{good_b:.1f}%"

        )


print(
    "\nFlow consistency and "
    "occlusion detection finished!"
)