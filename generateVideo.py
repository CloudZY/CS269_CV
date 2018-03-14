import cv2
import glob
import os

def imgs2video(imgs_dir, save_name):
    fps = 24
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (1024, 436))
    # no glob, need number-index increasing
    imgs = glob.glob(os.path.join(imgs_dir, '*.png'))

    images = []
    for image in os.listdir("./images_after_flow/"):
        images.append("./images_after_flow/" + str(image))
    #print(images)
    for i in images:
        frame = cv2.imread(i)
        video_writer.write(frame)

    video_writer.release()

imgs2video("./images_after_flow","video.avi")