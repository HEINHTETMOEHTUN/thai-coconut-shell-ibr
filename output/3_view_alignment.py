import cv2
import numpy as np
import glob
import os

rings = ["horizontal", "upper"]

sift = cv2.SIFT_create()
bf = cv2.BFMatcher()

for ring in rings:

    images = sorted(
        glob.glob(f"prepared/{ring}/*.JPG")
    )

    if len(images) < 2:
        print(f"Not enough images in {ring}")
        continue

    output_folder = f"aligned/{ring}"
    os.makedirs(output_folder, exist_ok=True)

    print(f"\nProcessing {ring}")
    print("Images found:", len(images))

    for i in range(len(images) - 1):

        img1 = cv2.imread(images[i])
        img2 = cv2.imread(images[i + 1])

        gray1 = cv2.cvtColor(
            img1,
            cv2.COLOR_BGR2GRAY
        )

        gray2 = cv2.cvtColor(
            img2,
            cv2.COLOR_BGR2GRAY
        )

        # SIFT features
        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            print(
                f"No descriptors: {ring} "
                f"view {i+1} -> view {i+2}"
            )
            continue

        # Match descriptors
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []

        for pair in matches:
            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            print(
                f"Not enough matches: {ring} "
                f"view {i+1} -> view {i+2}"
            )
            continue

        # Coordinates of matched points
        src_pts = np.float32(
            [kp2[m.trainIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        dst_pts = np.float32(
            [kp1[m.queryIdx].pt for m in good_matches]
        ).reshape(-1, 1, 2)

        # RANSAC homography
        H, mask = cv2.findHomography(
            src_pts,
            dst_pts,
            cv2.RANSAC,
            5.0
        )

        if H is None:
            print(
                f"Homography failed: {ring} "
                f"view {i+1} -> view {i+2}"
            )
            continue

        inliers = int(mask.sum())

        # Align second image to first image
        aligned_img = cv2.warpPerspective(
            img2,
            H,
            (img1.shape[1], img1.shape[0])
        )

        filename = (
            f"{output_folder}/"
            f"view_{i+2:02d}_aligned.jpg"
        )

        cv2.imwrite(filename, aligned_img)

        print(
            f"{ring}: view {i+1} -> view {i+2}: "
            f"{len(good_matches)} matches, "
            f"{inliers} RANSAC inliers"
        )

print("\nAll alignment finished.")