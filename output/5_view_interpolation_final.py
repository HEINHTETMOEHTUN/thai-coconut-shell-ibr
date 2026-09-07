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

ALPHAS = [
    0.25,
    0.50,
    0.75
]

PREPARED_ROOT = "prepared"

FLOW_ROOT = "optical_flows"

CONF_ROOT = "flow_confidence"

OUTPUT_ROOT = (
    "interpolated_frames"
)

DEBUG_ROOT = (
    "interpolation_debug"
)

LAPLACIAN_LEVELS = 5

LOW_CONFIDENCE_SUM = 0.08

EPS = 1e-6


# -------------------------------------------------
# COORDINATE GRID
# -------------------------------------------------

def make_grid(
    h,
    w
):

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

    return x, y


# -------------------------------------------------
# WARP IMAGE TOWARD OTHER VIEW
# -------------------------------------------------

def warp_image_and_validity(
    img,
    flow,
    fraction
):

    h, w = img.shape[:2]

    x, y = make_grid(
        h,
        w
    )

    # ---------------------------------------------
    # Approximate inverse warping
    #
    # destination samples:
    #
    # source position =
    # x - fraction * optical flow
    # ---------------------------------------------

    map_x = (

        x

        -

        fraction *
        flow[:, :, 0]

    )

    map_y = (

        y

        -

        fraction *
        flow[:, :, 1]

    )

    warped = cv2.remap(

        img,

        map_x,
        map_y,

        cv2.INTER_LINEAR,

        borderMode=(
            cv2.BORDER_CONSTANT
        ),

        borderValue=0
    )

    # ---------------------------------------------
    # VALIDITY MASK
    # ---------------------------------------------

    ones = np.ones(
        (
            h,
            w
        ),
        dtype=np.float32
    )

    valid = cv2.remap(

        ones,

        map_x,
        map_y,

        cv2.INTER_NEAREST,

        borderMode=(
            cv2.BORDER_CONSTANT
        ),

        borderValue=0
    )

    return (
        warped,
        valid,
        map_x,
        map_y
    )


# -------------------------------------------------
# WARP CONFIDENCE MAP
# -------------------------------------------------

def warp_scalar_map(
    value_map,
    map_x,
    map_y
):

    warped = cv2.remap(

        value_map.astype(
            np.float32
        ),

        map_x,
        map_y,

        cv2.INTER_LINEAR,

        borderMode=(
            cv2.BORDER_CONSTANT
        ),

        borderValue=0
    )

    return warped


# -------------------------------------------------
# LAPLACIAN PYRAMID BLENDING
# -------------------------------------------------

def laplacian_blend(
    img_a,
    img_b,
    mask_a,
    levels=5
):

    # Convert images to float

    a = (
        img_a.astype(
            np.float32
        )
        /
        255.0
    )

    b = (
        img_b.astype(
            np.float32
        )
        /
        255.0
    )

    mask = np.clip(

        mask_a.astype(
            np.float32
        ),

        0.0,
        1.0
    )

    h, w = mask.shape

    # Make sure pyramid
    # isn't too deep

    max_possible = max(

        1,

        int(
            np.floor(
                np.log2(
                    min(
                        h,
                        w
                    )
                )
            )
        )
        -
        2

    )

    levels = max(

        1,

        min(
            levels,
            max_possible
        )

    )

    # -------------------------------------------------
    # GAUSSIAN PYRAMIDS
    # -------------------------------------------------

    gp_a = [a]
    gp_b = [b]
    gp_m = [mask]

    for _ in range(
        levels
    ):

        gp_a.append(
            cv2.pyrDown(
                gp_a[-1]
            )
        )

        gp_b.append(
            cv2.pyrDown(
                gp_b[-1]
            )
        )

        gp_m.append(
            cv2.pyrDown(
                gp_m[-1]
            )
        )

    # -------------------------------------------------
    # LAPLACIAN PYRAMIDS
    # -------------------------------------------------

    lp_a = []
    lp_b = []

    for i in range(
        levels
    ):

        size_a = (
            gp_a[i].shape[1],
            gp_a[i].shape[0]
        )

        size_b = (
            gp_b[i].shape[1],
            gp_b[i].shape[0]
        )

        up_a = cv2.pyrUp(

            gp_a[i + 1],

            dstsize=size_a
        )

        up_b = cv2.pyrUp(

            gp_b[i + 1],

            dstsize=size_b
        )

        lp_a.append(

            gp_a[i]
            -
            up_a

        )

        lp_b.append(

            gp_b[i]
            -
            up_b

        )

    lp_a.append(
        gp_a[-1]
    )

    lp_b.append(
        gp_b[-1]
    )

    # -------------------------------------------------
    # BLEND PYRAMIDS
    # -------------------------------------------------

    blended_levels = []

    for (
        lap_a,
        lap_b,
        gaussian_mask
    ) in zip(

        lp_a,
        lp_b,
        gp_m

    ):

        mask_3d = (
            gaussian_mask[
                :,
                :,
                None
            ]
        )

        blended = (

            lap_a *
            mask_3d

            +

            lap_b *
            (
                1.0 -
                mask_3d
            )

        )

        blended_levels.append(
            blended
        )

    # -------------------------------------------------
    # RECONSTRUCT IMAGE
    # -------------------------------------------------

    result = (
        blended_levels[-1]
    )

    for i in range(
        levels - 1,
        -1,
        -1
    ):

        size = (

            blended_levels[i].shape[1],

            blended_levels[i].shape[0]

        )

        result = cv2.pyrUp(

            result,

            dstsize=size

        )

        result = (

            result

            +

            blended_levels[i]

        )

    result = np.clip(

        result * 255.0,

        0,
        255

    ).astype(
        np.uint8
    )

    return result


# -------------------------------------------------
# PROCESS EACH RING
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

    output_folder = os.path.join(
        OUTPUT_ROOT,
        ring
    )

    debug_folder = os.path.join(
        DEBUG_ROOT,
        ring
    )

    if os.path.exists(
        output_folder
    ):

        shutil.rmtree(
            output_folder
        )

    if os.path.exists(
        debug_folder
    ):

        shutil.rmtree(
            debug_folder
        )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    os.makedirs(
        debug_folder,
        exist_ok=True
    )

    print(
        f"\nImproved interpolation: "
        f"{ring}"
    )

    # -------------------------------------------------
    # EACH NEIGHBORING PAIR
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

        # -------------------------------------------------
        # FILE PATHS
        # -------------------------------------------------

        forward_path = os.path.join(

            FLOW_ROOT,
            ring,

            f"flow_fwd_{i:02d}.npy"

        )

        backward_path = os.path.join(

            FLOW_ROOT,
            ring,

            f"flow_bwd_{i:02d}.npy"

        )

        conf_a_path = os.path.join(

            CONF_ROOT,
            ring,

            f"conf_a_{i:02d}.npy"

        )

        conf_b_path = os.path.join(

            CONF_ROOT,
            ring,

            f"conf_b_{i:02d}.npy"

        )

        required_files = [

            forward_path,
            backward_path,
            conf_a_path,
            conf_b_path

        ]

        if not all(

            os.path.exists(path)

            for path
            in required_files

        ):

            print(

                f"Missing flow/"
                f"confidence files "
                f"for pair {i:02d}"

            )

            continue

        # -------------------------------------------------
        # LOAD FLOW + CONFIDENCE
        # -------------------------------------------------

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

        conf_a = np.load(
            conf_a_path
        ).astype(
            np.float32
        )

        conf_b = np.load(
            conf_b_path
        ).astype(
            np.float32
        )

        if (

            flow_fwd.shape[:2]

            !=

            img_a.shape[:2]

        ):

            print(

                f"Flow size mismatch "
                f"for pair {i:02d}"

            )

            continue

        # -------------------------------------------------
        # GENERATE INTERMEDIATE VIEWS
        # -------------------------------------------------

        for step, alpha in enumerate(

            ALPHAS,
            start=1

        ):

            # =============================================
            # WARP A TOWARD B
            # =============================================

            (
                warp_a,
                valid_a,
                map_ax,
                map_ay

            ) = warp_image_and_validity(

                img_a,

                flow_fwd,

                alpha

            )

            # =============================================
            # WARP B TOWARD A
            # =============================================

            (
                warp_b,
                valid_b,
                map_bx,
                map_by

            ) = warp_image_and_validity(

                img_b,

                flow_bwd,

                1.0 - alpha

            )

            # =============================================
            # MOVE CONFIDENCE MAPS
            # INTO INTERMEDIATE POSITION
            # =============================================

            conf_a_warp = (

                warp_scalar_map(

                    conf_a,

                    map_ax,
                    map_ay

                )

                *

                valid_a
            )

            conf_b_warp = (

                warp_scalar_map(

                    conf_b,

                    map_bx,
                    map_by

                )

                *

                valid_b
            )

            # =============================================
            # TEMPORAL + CONFIDENCE WEIGHTS
            # =============================================

            weight_a = (

                (1.0 - alpha)

                *

                conf_a_warp

            )

            weight_b = (

                alpha

                *

                conf_b_warp

            )

            weight_sum = (

                weight_a
                +
                weight_b

            )

            # =============================================
            # NORMALIZED BLENDING MASK
            # =============================================

            mask_a = np.zeros_like(

                weight_sum,

                dtype=np.float32

            )

            reliable = (

                weight_sum
                >
                EPS

            )

            mask_a[
                reliable
            ] = (

                weight_a[
                    reliable
                ]

                /

                weight_sum[
                    reliable
                ]

            )

            # =============================================
            # LOW CONFIDENCE:
            # DON'T BLEND TWO BAD PIXELS
            # =============================================

            low_confidence = (

                weight_sum

                <

                LOW_CONFIDENCE_SUM

            )

            choose_a = (

                weight_a
                >=
                weight_b

            )

            # If both are almost zero,
            # use nearest temporal view.

            both_zero = (

                weight_sum
                <=
                EPS

            )

            choose_a[
                both_zero
            ] = (

                alpha <= 0.5
            )

            mask_a[

                low_confidence
                &
                choose_a

            ] = 1.0

            mask_a[

                low_confidence
                &
                (~choose_a)

            ] = 0.0

            # =============================================
            # INVALID WARP AREAS
            # =============================================

            only_a_valid = (

                (valid_a > 0.5)

                &

                (valid_b <= 0.5)

            )

            only_b_valid = (

                (valid_b > 0.5)

                &

                (valid_a <= 0.5)

            )

            mask_a[
                only_a_valid
            ] = 1.0

            mask_a[
                only_b_valid
            ] = 0.0

            # =============================================
            # LAPLACIAN PYRAMID BLENDING
            # =============================================

            result = laplacian_blend(

                warp_a,

                warp_b,

                mask_a,

                levels=(
                    LAPLACIAN_LEVELS
                )

            )

            # =============================================
            # HARD PROTECTION FOR VERY BAD AREAS
            # =============================================

            very_low = (

                weight_sum

                <

                (
                    LOW_CONFIDENCE_SUM
                    *
                    0.5
                )

            )

            select_a = (

                very_low
                &
                choose_a

            )

            select_b = (

                very_low
                &
                (~choose_a)

            )

            # Instead of blending,
            # directly select reliable view.

            result[
                select_a
            ] = warp_a[
                select_a
            ]

            result[
                select_b
            ] = warp_b[
                select_b
            ]

            # =============================================
            # SAVE FINAL INTERPOLATED IMAGE
            # =============================================

            filename = os.path.join(

                output_folder,

                f"pair_{i:02d}_"
                f"step_{step:02d}.jpg"

            )

            cv2.imwrite(
                filename,
                result
            )

            # =============================================
            # SAVE DEBUG IMAGES FOR ALPHA 0.5
            # =============================================

            if abs(
                alpha - 0.50
            ) < 1e-6:

                cv2.imwrite(

                    os.path.join(

                        debug_folder,

                        f"pair_{i:02d}_"
                        f"warp_a.jpg"

                    ),

                    warp_a

                )

                cv2.imwrite(

                    os.path.join(

                        debug_folder,

                        f"pair_{i:02d}_"
                        f"warp_b.jpg"

                    ),

                    warp_b

                )

                mask_image = np.clip(

                    mask_a * 255,

                    0,
                    255

                ).astype(
                    np.uint8
                )

                cv2.imwrite(

                    os.path.join(

                        debug_folder,

                        f"pair_{i:02d}_"
                        f"blend_mask_a.jpg"

                    ),

                    mask_image

                )

                confidence_sum_image = (

                    cv2.normalize(

                        weight_sum,

                        None,

                        0,
                        255,

                        cv2.NORM_MINMAX

                    ).astype(
                        np.uint8
                    )

                )

                cv2.imwrite(

                    os.path.join(

                        debug_folder,

                        f"pair_{i:02d}_"
                        f"confidence_sum.jpg"

                    ),

                    confidence_sum_image

                )

            print(

                f"{ring}: "

                f"view {i+1}"
                f"->"
                f"{i+2}, "

                f"alpha="
                f"{alpha:.2f}"

            )


print(
    "\nAll improved "
    "interpolated views finished!"
)