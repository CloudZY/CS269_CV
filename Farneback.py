import cv2
import numpy as np
import os
import flow_visualization
import evaluation

def draw_flow(img, flow, step=16):
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2 : h : step, step/2 : w : step].reshape(2, -1)
    fx, fy = flow[y, x].T
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.polylines(vis, lines, 0, (0, 255, 0))
    for (x1, y1), (x2, y2) in lines:
        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    return vis

def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:,:,0], flow[:,:,1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx*fx + fy*fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[...,0] = ang * (180 / np.pi / 2)
    hsv[...,1] = 255
    hsv[...,2] = np.minimum(v*4, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:,:,0] += np.arange(w)
    flow[:,:,1] += np.arange(h)[:,np.newaxis]
    res = cv2.remap(img, flow, None, cv2.INTER_LINEAR)
    return res

def gunnar_farneback(img1, img2):
    img1_gray = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    img2_gray = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)
    flow = cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, 0.5, 3, 15, 3, 5, 2, 0)
    return flow

if __name__ == '__main__':
    # image_directory_path = './other-data'
    # ground_truth_directory_path = './other-gt-flow'
    # dataset = []
    # epe_set = []
    # aae_set = []
    # for dataset_name in os.listdir(image_directory_path):
    #     dataset.append(dataset_name)
    #     predict_flow = gunnar_farneback(image_directory_path+'/'+dataset_name+'/frame10.png',
    #                         image_directory_path + '/' + dataset_name + '/frame11.png')
    #     ground_flow = evaluation.read_flow(ground_truth_directory_path+'/'+dataset_name+'/flow10.flo')
    #     epe = evaluation.EPE_RB(ground_flow, predict_flow)
    #     aae = evaluation.AAE_RB(ground_flow, predict_flow)
    #     epe_set.append(epe)
    #     aae_set.append(aae)
    # print(dataset)
    # print(epe_set)
    # print(aae_set)
    #
    # with open('Gunnar_Farneback_result.txt', 'w') as f:
    #     for i in range(len(dataset)):
    #         f.write(dataset[i] + ' ' + str(epe_set[i]) + ' ' + str(aae_set[i]) + '\n')
    # f.close()

# if __name__ == '__main__':
#     # Optical flow for 2 images
#     # prev = cv2.imread('./other-data-gray/Beanbags/frame10.png')
#     # curr = cv2.imread('./other-data-gray/Beanbags/frame11.png')
#     #
#     # prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#     # curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
#     # h, w = prev.shape[:2]
#     # flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, 0.5, 3, 15, 3, 5, 1.2, 0)
#     #
#     # cv2.imshow('flow', draw_flow(curr_gray, flow))
#     # cv2.imshow('flow HSV', draw_hsv(flow))
#     # cv2.imshow('glitch', warp_flow(curr, flow))
#     # cv2.waitKey(0)
#     # cv2.destroyAllWindows()
#
#     cam = cv2.VideoCapture('')
#     ret, prev = cam.read()
#     prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
#     show_hsv = False
#     show_glitch = False
#     cur_glitch = prev.copy()
#
#     while True:
#         ret, img = cam.read()
#         if ret == True:
#             if cv2.cv.WaitKey(10) == 27:
#                 break
#             gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#             flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, 0.5, 3, 15, 3, 5, 1.2, 0)
#             prev_gray = gray
#             cv2.imshow('flow', draw_flow(gray, flow))
#             if show_hsv:
#                 cv2.imshow('flow HSV', draw_hsv(flow))
#             if show_glitch:
#                 cur_glitch = warp_flow(cur_glitch, flow)
#                 cv2.imshow('glitch', cur_glitch)
#
#             # ch = 0xFF & cv2.waitKey(5)
#             # if ch == 27:
#             #     break
#             # if ch == ord('1'):
#             #     show_hsv = not show_hsv
#             #     print 'HSV flow visualization is', ['off', 'on'][show_hsv]
#             # if ch == ord('2'):
#             #     show_glitch = not show_glitch
#             #     if show_glitch:
#             #         cur_glitch = img.copy()
#             #     print 'glitch is', ['off', 'on'][show_glitch]
#         else:
#             break
#         cv2.destroyAllWindows()

    images = []
    outnames = []

    for image in os.listdir("./images/"):
        images.append("./images/" + str(image))
    images = images[:-1]
    for i in range(len(images)):
        name = str(i+1)
        if i+1 < 10:
            name = "0" + name
        outnames.append("./images_after_flow_farneback/" + "out_image_" + name + ".png")
    for i in range(len(images)):
        if i+1 == len(images):
            break
        print(images[i],images[i+1])
        flow_visualization.save_flow_farneback(gunnar_farneback(images[i],images[i+1]), images[i], outnames[i])