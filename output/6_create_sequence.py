import cv2
import glob
import os
import shutil

rings = ["horizontal", "upper"]

for ring in rings:

    real_images = sorted(
        glob.glob(f"prepared/{ring}/*.JPG")
    )

    if len(real_images) < 2:
        print(f"Not enough real images in {ring}")
        continue

    output_folder = f"final_sequence/{ring}"

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    count = 0

    print(f"\nCreating sequence for {ring}")

    for i in range(len(real_images) - 1):

        # Add current real image
        img = cv2.imread(real_images[i])

        filename = (
            f"{output_folder}/frame_{count:03d}.jpg"
        )

        cv2.imwrite(filename, img)
        count += 1

        # Add the 3 interpolated images
        generated = sorted(
            glob.glob(
                f"interpolated_frames/{ring}/"
                f"pair_{i:02d}_step_*.jpg"
            )
        )

        for path in generated:

            img = cv2.imread(path)

            filename = (
                f"{output_folder}/frame_{count:03d}.jpg"
            )

            cv2.imwrite(filename, img)
            count += 1

    # Add the final real image
    last_img = cv2.imread(real_images[-1])

    filename = (
        f"{output_folder}/frame_{count:03d}.jpg"
    )

    cv2.imwrite(filename, last_img)
    count += 1

    print(
        f"{ring} sequence complete."
    )

    print(
        f"Total frames: {count}"
    )

print("\nAll sequences created!")