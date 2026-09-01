import cv2
import torch
import numpy as np
import os

# -----------------------------
# 1. Load MiDaS depth model
# -----------------------------

model_type = "MiDaS_small"

midas = torch.hub.load(
    "intel-isl/MiDaS",
    model_type
)

midas.eval()

# Load MiDaS image transformations
transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = transforms.small_transform


# -----------------------------
# 2. Load owl image
# -----------------------------

image_path = "prepared/view01.JPG"

img = cv2.imread(image_path)

if img is None:
    print("Image not found:", image_path)
    exit()

# OpenCV uses BGR
# MiDaS expects RGB

img_rgb = cv2.cvtColor(
    img,
    cv2.COLOR_BGR2RGB
)


# -----------------------------
# 3. Prepare image for model
# -----------------------------

input_batch = transform(img_rgb)


# -----------------------------
# 4. Estimate depth
# -----------------------------

with torch.no_grad():

    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False
    ).squeeze()


depth = prediction.cpu().numpy()


# -----------------------------
# 5. Convert depth to image
# -----------------------------

depth_normalized = cv2.normalize(
    depth,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

depth_normalized = depth_normalized.astype(
    np.uint8
)


# -----------------------------
# 6. Save depth map
# -----------------------------

cv2.imwrite(
    "depth_map.jpg",
    depth_normalized
)

print("Depth map created!")
print("Saved as depth_map.jpg")


# -----------------------------
# 7. Display result
# -----------------------------

cv2.imshow(
    "Original Coconut Handicraft",
    img
)

cv2.imshow(
    "Estimated Depth Map",
    depth_normalized
)

cv2.waitKey(0)
cv2.destroyAllWindows()