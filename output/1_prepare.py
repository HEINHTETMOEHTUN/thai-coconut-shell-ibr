import cv2
import os
import glob
import shutil

rings = ["horizontal", "upper"]

for ring in rings:

    input_folder = f"../images/{ring}"
    output_folder = f"prepared/{ring}"

    # Remove old prepared images for this ring
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

    os.makedirs(output_folder, exist_ok=True)

    files = sorted(glob.glob(f"{input_folder}/*"))

    print(f"{ring}: {len(files)} images found")

    for i, filename in enumerate(files):

        img = cv2.imread(filename)

        if img is None:
            print("Could not read:", filename)
            continue

        img = cv2.resize(img, (800, 800))

        output = f"{output_folder}/view{i+1:02d}.JPG"

        cv2.imwrite(output, img)

        print("Prepared:", output)

print("All images prepared!")