import cv2
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

    output_folder = f"feature_matches/{ring}"
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

        kp1, des1 = sift.detectAndCompute(gray1, None)
        kp2, des2 = sift.detectAndCompute(gray2, None)

        if des1 is None or des2 is None:
            print(
                f"No descriptors: view {i+1} -> view {i+2}"
            )
            continue

        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []

        for pair in matches:

            if len(pair) < 2:
                continue

            m, n = pair

            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        result = cv2.drawMatches(
            img1,
            kp1,
            img2,
            kp2,
            good_matches,
            None,
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        )

        filename = (
            f"{output_folder}/match_{i:02d}.jpg"
        )

        cv2.imwrite(filename, result)

        print(
            f"{ring}: view {i+1} -> view {i+2}: "
            f"{len(good_matches)} good matches"
        )

print("\nAll feature matching finished.")