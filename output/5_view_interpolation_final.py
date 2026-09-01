import cv2
import numpy as np
import glob
import os
import shutil

rings = ["horizontal", "upper"]

alphas = [0.25, 0.50, 0.75]

for ring in rings:

    images = sorted(
        glob.glob(f"prepared/{ring}/*.JPG")
    )

    flows = sorted(
        glob.glob(f"optical_flows/{ring}/*.npy")
    )

    if len(images) < 2:
        print(f"Not enough images in {ring}")
        continue

    if len(flows) != len(images) - 1:
        print(f"Flow count does not match images in {ring}")
        print("Images:", len(images))
        print("Flows:", len(flows))
        continue

    output_folder = f"interpolated_frames/{ring}"

    # Remove old interpolated results
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    print(f"\nProcessing {ring}")
    print("Images found:", len(images))
    print("Flows found:", len(flows))

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

        flow = np.load(flows[i])

        h, w = img1.shape[:2]

        x, y = np.meshgrid(
            np.arange(w),
            np.arange(h)
        )

        for step, alpha in enumerate(alphas, 1):

            # Warp first image toward second
            map_x1 = (
                x + flow[:, :, 0] * alpha
            ).astype(np.float32)

            map_y1 = (
                y + flow[:, :, 1] * alpha
            ).astype(np.float32)

            # Warp second image backward
            map_x2 = (
                x - flow[:, :, 0] * (1 - alpha)
            ).astype(np.float32)

            map_y2 = (
                y - flow[:, :, 1] * (1 - alpha)
            ).astype(np.float32)

            warp1 = cv2.remap(
                img1,
                map_x1,
                map_y1,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

            warp2 = cv2.remap(
                img2,
                map_x2,
                map_y2,
                cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT
            )

            # Blend both warped images
            result = cv2.addWeighted(
                warp1,
                1 - alpha,
                warp2,
                alpha,
                0
            )

            filename = (
                f"{output_folder}/"
                f"pair_{i:02d}_step_{step:02d}.jpg"
            )

            cv2.imwrite(filename, result)

            print(
                f"{ring}: view {i+1}->{i+2}, "
                f"alpha={alpha:.2f}"
            )

print("\nAll interpolated views finished!")