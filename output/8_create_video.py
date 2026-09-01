import cv2
import glob


def create_video(frame_folder, output_name, fps=8):

    frames = sorted(
        glob.glob(f"{frame_folder}/*.jpg")
    )

    if len(frames) == 0:
        print(f"No frames found in {frame_folder}")
        return

    first = cv2.imread(frames[0])

    if first is None:
        print(f"Could not read first frame in {frame_folder}")
        return

    height, width = first.shape[:2]

    # MJPG is more reliable on Mac than mp4v
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    video = cv2.VideoWriter(
        output_name,
        fourcc,
        fps,
        (width, height)
    )

    if not video.isOpened():
        print(f"Could not create {output_name}")
        return

    for filename in frames:

        frame = cv2.imread(filename)

        if frame is None:
            print("Could not read:", filename)
            continue

        frame = cv2.resize(
            frame,
            (width, height)
        )

        video.write(frame)

    video.release()

    print(
        f"Created: {output_name} "
        f"({len(frames)} frames)"
    )


# -----------------------------------
# 1. Horizontal rendering video
# -----------------------------------

create_video(
    "final_sequence/horizontal",
    "horizontal_rendering.avi",
    fps=8
)


# -----------------------------------
# 2. Upper rendering video
# -----------------------------------

create_video(
    "final_sequence/upper",
    "upper_rendering.avi",
    fps=8
)


# -----------------------------------
# 3. Combined horizontal + upper
# -----------------------------------

horizontal_frames = sorted(
    glob.glob("final_sequence/horizontal/*.jpg")
)

upper_frames = sorted(
    glob.glob("final_sequence/upper/*.jpg")
)

all_frames = horizontal_frames + upper_frames

if len(all_frames) == 0:

    print("No frames found for combined video.")

else:

    first = cv2.imread(all_frames[0])

    if first is None:

        print("Could not read first combined frame.")

    else:

        height, width = first.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")

        video = cv2.VideoWriter(
            "owl_combined_rendering.avi",
            fourcc,
            8,
            (width, height)
        )

        if not video.isOpened():

            print("Could not create combined video.")

        else:

            for filename in all_frames:

                frame = cv2.imread(filename)

                if frame is None:
                    print("Could not read:", filename)
                    continue

                frame = cv2.resize(
                    frame,
                    (width, height)
                )

                video.write(frame)

            video.release()

            print(
                f"Created: owl_combined_rendering.avi "
                f"({len(all_frames)} frames)"
            )


print("All videos finished!")