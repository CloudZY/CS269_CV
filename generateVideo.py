import cv2
import glob
import os

def imgs2video(imgs_dir, save_name):
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (1920, 1080))
    # no glob, need number-index increasing
    imgs = glob.glob(os.path.join(imgs_dir, '*.png'))

    for i in range(len(imgs)):
        imgname = os.path.join(imgs_dir, 'core-{:02d}.png'.format(i))
        frame = cv2.imread(imgname)
        video_writer.write(frame)

    video_writer.release()

imgs2video("./images_after_flow","video")