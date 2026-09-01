import cv2
import os
import numpy as np

folder = "prepared"

files = sorted([
    f for f in os.listdir(folder)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

images = []

for file in files:
    img = cv2.imread(os.path.join(folder, file))

    if img is not None:
        img = cv2.resize(img, (250, 350))
        images.append(img)

# Show 4 images in each row
rows = []

for i in range(0, len(images), 4):
    row = images[i:i+4]

    # Fill incomplete final row
    while len(row) < 4:
        row.append(np.zeros_like(images[0]))

    rows.append(np.hstack(row))

light_field = np.vstack(rows)

cv2.imwrite("light_field.jpg", light_field)

cv2.imshow("Light Field - Coconut Handicraft", light_field)
cv2.waitKey(0)
cv2.destroyAllWindows()