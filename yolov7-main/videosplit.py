import cv2
import os


def split(video):
    # Set save directory name
    save_dir = 'test_video'

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    # Get the total frames of video
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)

    # 当前帧数
    current_frame = 0

    while True:
        # Get current frame
        _, frame = video.read()

        # If no frame exists, exit
        if not _:
            break

        zfill = len(str(int(total_frames)))

        # Save every frame
        filename = os.path.join(save_dir, "frame%s.jpg" % str(current_frame).zfill(zfill))

        cv2.imwrite(filename, frame)
        print('Saved', filename)

        # Next frame
        current_frame += 1

    # Release video object
    video.release()


if __name__ == '__main__':
    path = 'test_video.mp4'
    video = cv2.VideoCapture(path)
    split(video)
