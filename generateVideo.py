import cv2
import glob
import os

# File that use to generate and store the video using images

# imgs_dir : directory of all images that need to convert to vedio
# save_name : name of the generated vedio
def imgs2video(imgs_dir, save_name):
    fps = 6
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    video_writer = cv2.VideoWriter(save_name, fourcc, fps, (1024, 436))
    # no glob, need number-index increasing
    imgs = glob.glob(os.path.join(imgs_dir, '*.png'))

    images = []
    for image in os.listdir(imgs_dir):
        images.append(imgs_dir + str(image))
    #print(images)
    for i in images:
        frame = cv2.imread(i)
        video_writer.write(frame)

    video_writer.release()

if __name__ == '__main__':
    # Convert all images to vedio
    imgs2video("./images_after_warp_farneback","farneback_after_warp_video.avi")