import cv2
import glob
import os

from tqdm import tqdm


def merge():
    image_folder = 'vis/'

    # Set video parameters
    fps = 25
    frame_size = (1910, 1080)
    video_name = 'demo.mp4'

    # Get images list
    images = glob.glob(os.path.join(image_folder, '*.jpg'))
    images.sort()

    videoWriter = cv2.VideoWriter(video_name,
                                  cv2.VideoWriter_fourcc(*'MP4V'),
                                  fps, frame_size)

    for image in tqdm(images):
        img = cv2.imread(image)

        videoWriter.write(img)

    videoWriter.release()


if __name__ == '__main__':
    merge()
