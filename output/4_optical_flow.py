import cv2
import numpy as np
import glob
import os
import shutil

rings = ["horizontal", "upper"]

for ring in rings:

    images = sorted(
        glob.glob(f"prepared/{ring}/*.JPG")
    )

    if len(images) < 2:
        print(f"Not enough images in {ring}")
        continue

    output_folder = f"optical_flows/{ring}"

    # Remove old flow files for this ring
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    print(f"\nProcessing {ring}")
    print("Images found:", len(images))

    for i in range(len(images) - 1):

        img1 = cv2.imread(images[i])
        img2 = cv2.imread(images[i + 1])

        if img1 is None or img2 is None:
            print(
                f"Could not read {ring} "
                f"view {i+1} or view {i+2}"
            )
            continue

        img2 = cv2.resize(
            img2,
            (img1.shape[1], img1.shape[0])
        )

        gray1 = cv2.cvtColor(
            img1,
            cv2.COLOR_BGR2GRAY
        )

        gray2 = cv2.cvtColor(
            img2,
            cv2.COLOR_BGR2GRAY
        )

        flow = cv2.calcOpticalFlowFarneback(
            gray1,
            gray2,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )

        filename = (
            f"{output_folder}/flow_{i:02d}.npy"
        )

        np.save(filename, flow)

        print(
            f"{ring}: view {i+1} -> view {i+2}"
        )

print("\nAll optical flows finished!")