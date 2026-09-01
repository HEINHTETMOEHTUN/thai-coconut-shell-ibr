import cv2
import os

folder = "prepared"

img1 = cv2.imread(os.path.join(folder, "view01.JPG"))
img2 = cv2.imread(os.path.join(folder, "view02.JPG"))

if img1 is None or img2 is None:
    print("Images not found.")
    exit()

# Make sure both images have the same size
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

alphas = [0.25, 0.50, 0.75]

for alpha in alphas:
    result = cv2.addWeighted(
        img1, 1 - alpha,
        img2, alpha,
        0
    )

    cv2.imwrite(
        f"interpolation_{alpha}.jpg",
        result
    )

    cv2.imshow(
        f"Interpolation {alpha}",
        result
    )

cv2.imshow("View 1", img1)
cv2.imshow("View 2", img2)

cv2.waitKey(0)
cv2.destroyAllWindows()

print("Interpolation finished!")