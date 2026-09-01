import cv2
import glob

horizontal_frames = sorted(
    glob.glob("final_sequence/horizontal/*.jpg")
)

upper_frames = sorted(
    glob.glob("final_sequence/upper/*.jpg")
)

if len(horizontal_frames) == 0 or len(upper_frames) == 0:
    print("Final sequences not found.")
    exit()

mode = "horizontal"
index = 0

while True:

    if mode == "horizontal":
        frames = horizontal_frames
    else:
        frames = upper_frames

    if index >= len(frames):
        index = len(frames) - 1

    img = cv2.imread(frames[index])

    cv2.putText(
        img,
        f"{mode.upper()}  View {index + 1}/{len(frames)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.imshow(
        "Thai Coconut-Shell Owl Rendering",
        img
    )

    key = cv2.waitKey(0) & 0xFF

    # Next viewpoint
    if key == ord("d"):
        index = (index + 1) % len(frames)

    # Previous viewpoint
    elif key == ord("a"):
        index = (index - 1) % len(frames)

    # Horizontal mode
    elif key == ord("h"):
        mode = "horizontal"
        index = 0

    # Upper mode
    elif key == ord("u"):
        mode = "upper"
        index = 0

    # Quit
    elif key == ord("q"):
        break

cv2.destroyAllWindows()