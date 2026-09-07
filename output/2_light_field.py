import cv2
import os
import glob
import numpy as np


RINGS = [
    "horizontal",
    "upper"
]

PREPARED_ROOT = "prepared"

OUTPUT_ROOT = "light_fields"

THUMBNAIL_SIZE = (
    200,
    200
)

IMAGES_PER_ROW = 4


os.makedirs(
    OUTPUT_ROOT,
    exist_ok=True
)


for ring in RINGS:

    files = sorted(
        glob.glob(
            os.path.join(
                PREPARED_ROOT,
                ring,
                "*.JPG"
            )
        )
    )

    if len(files) == 0:

        print(
            f"No prepared images "
            f"found in {ring}"
        )

        continue

    images = []

    for file in files:

        img = cv2.imread(
            file
        )

        if img is None:

            continue

        img = cv2.resize(
            img,
            THUMBNAIL_SIZE,
            interpolation=cv2.INTER_AREA
        )

        images.append(
            img
        )

    if len(images) == 0:

        print(
            f"No readable images "
            f"in {ring}"
        )

        continue

    rows = []

    for i in range(
        0,
        len(images),
        IMAGES_PER_ROW
    ):

        row = images[
            i:
            i + IMAGES_PER_ROW
        ]

        while (
            len(row) <
            IMAGES_PER_ROW
        ):

            row.append(
                np.zeros_like(
                    images[0]
                )
            )

        rows.append(
            np.hstack(row)
        )

    light_field = np.vstack(
        rows
    )

    output_path = os.path.join(
        OUTPUT_ROOT,
        f"{ring}_light_field.jpg"
    )

    cv2.imwrite(
        output_path,
        light_field
    )

    cv2.imshow(
        f"Light Field - {ring}",
        light_field
    )

    print(
        "Saved:",
        output_path
    )


cv2.waitKey(0)

cv2.destroyAllWindows()

print(
    "All light-field grids finished!"
)