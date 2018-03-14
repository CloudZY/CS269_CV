import cv
import numpy as np
import os
import evaluation
FLOWSKIP = 16

def draw_flow(img, velx, vely, step=FLOWSKIP):
    cols = img.width
    rows = img.height
    for i in range(0, cols, step):
        for j in range(0, rows, step):
            dx = int(cv.GetReal2D(velx, j, i))
            dy = int(cv.GetReal2D(vely, j, i))
            cv.Line(img, (i, j), (i + dx, j + dy), (0, 255, 0))
            cv.Circle(img, (i, j), 1, (0, 255, 0), -1)
    return img

def draw_hsv(velx, vely):
    cols = velx.width
    rows = velx.height
    ang = np.arctan2(vely, velx) + np.pi
    v = np.zeros([rows, cols])
    for i in range(0, cols):
        for j in range(0, rows):
            v[j][i] = np.sqrt(cv.GetReal2D(velx, j, i)*cv.GetReal2D(velx, j, i) + cv.GetReal2D(vely, j, i)*cv.GetReal2D(vely, j, i))
    hsv = np.zeros((rows, cols, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = 255
    hsv[..., 2] = np.minimum(v * 4, 255)
    bgr = cv.fromarray(np.zeros((rows, cols, 3), np.uint8))
    cv.CvtColor(cv.fromarray(hsv), bgr, cv.CV_HSV2BGR)
    return bgr

def warp_flow(img, velx, vely):
    cols = img.width
    rows = img.height
    vel_zero = cv.CreateMat(rows, cols, cv.CV_32FC1)
    cv.Sub(vel_zero, velx, velx)
    cv.Sub(vel_zero, vely, vely)
    for i in range(0, cols):
        for j in range(0, rows):
            cv.SetReal2D(velx, j, i, cv.GetReal2D(velx, j, i) + i)
            cv.SetReal2D(vely, j, i, cv.GetReal2D(vely, j, i) + j)
    res = img
    cv.Remap(img, res, velx, vely, flags=cv.CV_INTER_LINEAR)
    return res

def horn_schunck(img1, img2):
    img1_gray = cv.LoadImage(img1, cv.CV_LOAD_IMAGE_GRAYSCALE)
    img2_gray = cv.LoadImage(img2, cv.CV_LOAD_IMAGE_GRAYSCALE)
    cols = img1_gray.width
    rows = img1_gray.height
    # initial the flow matrix
    velx = cv.CreateMat(rows, cols, cv.CV_32FC1)
    vely = cv.CreateMat(rows, cols, cv.CV_32FC1)
    cv.SetZero(velx)
    cv.SetZero(vely)
    cv.CalcOpticalFlowHS(img1_gray, img2_gray, False, velx, vely, 100.0,
                         (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 64, 0.01))

    res = np.zeros([rows, cols, 2])
    for i in range(rows):
        for j in range(cols):
            res[i][j][0] = cv.GetReal2D(velx, i, j)
            res[i][j][1] = cv.GetReal2D(vely, i, j)

    return res

def lucas_kanade(img1, img2):
    img1_gray = cv.LoadImage(img1, cv.CV_LOAD_IMAGE_GRAYSCALE)
    img2_gray = cv.LoadImage(img2, cv.CV_LOAD_IMAGE_GRAYSCALE)
    cols = img1_gray.width
    rows = img1_gray.height
    # initial the flow matrix
    velx = cv.CreateMat(rows, cols, cv.CV_32FC1)
    vely = cv.CreateMat(rows, cols, cv.CV_32FC1)
    cv.SetZero(velx)
    cv.SetZero(vely)
    cv.CalcOpticalFlowLK(img1_gray, img2_gray, (15, 15), velx, vely)

    res = np.zeros([rows, cols, 2])
    for i in range(rows):
        for j in range(cols):
            res[i][j][0] = cv.GetReal2D(velx, i, j)
            res[i][j][1] = cv.GetReal2D(vely, i, j)

    return res

# if __name__ == '__main__':
#     prev_gray = cv.LoadImage('./other-data-gray/Beanbags/frame10.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
#     curr_gray = cv.LoadImage('./other-data-gray/Beanbags/frame11.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
#
#     prev_color = cv.LoadImage('./other-data-gray/Beanbags/frame10.png', cv.CV_LOAD_IMAGE_COLOR)
#     curr_color = cv.LoadImage('./other-data-gray/Beanbags/frame11.png', cv.CV_LOAD_IMAGE_COLOR)
#
#     cols = prev_gray.width
#     rows = prev_gray.height
#
#     velx = cv.CreateMat(rows, cols, cv.CV_32FC1)
#     vely = cv.CreateMat(rows, cols, cv.CV_32FC1)
#     cv.SetZero(velx)
#     cv.SetZero(vely)
#
#     cv.CalcOpticalFlowHS(prev_gray, curr_gray, False, velx, vely, 100.0,
#                          (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 64, 0.01))
#
#     # cv.CalcOpticalFlowLK(prev_gray, curr_gray, (15, 15), velx, vely)
#
#     cv.NamedWindow("Optical flow HS")
#     cv.ShowImage("Optical flow HS", draw_flow(prev_color, velx, vely))
#     # cv.SaveImage("resultHS.png", desImageHS)
#
#     cv.NamedWindow("flow HSV")
#     cv.ShowImage("flow HSV", draw_hsv(velx, vely))
#
#     cv.NamedWindow("glitch")
#     cv.ShowImage("glitch", warp_flow(prev_color, velx, vely))
#
#     cv.WaitKey(0)
#     cv.DestroyAllWindows()

if __name__ == '__main__':
    image_directory_path = './other-data'
    ground_truth_directory_path = './other-gt-flow'
    dataset = []
    epe_set = []
    aae_set = []
    for dataset_name in os.listdir(image_directory_path):
        dataset.append(dataset_name)
        predict_flow = lucas_kanade(image_directory_path+'/'+dataset_name+'/frame10.png',
                            image_directory_path + '/' + dataset_name + '/frame11.png')
        ground_flow = evaluation.read_flow(ground_truth_directory_path+'/'+dataset_name+'/flow10.flo')
        epe = evaluation.EPE_RB(ground_flow, predict_flow)
        aae = evaluation.AAE_RB(ground_flow, predict_flow)
        epe_set.append(epe)
        aae_set.append(aae)
    print(dataset)
    print(epe_set)
    print(aae_set)

    with open('Lucas_Kanade_result.txt', 'w') as f:
    # with open('Horn_Schunck_result.txt', 'w') as f:
        for i in range(len(dataset)):
            f.write(dataset[i] + ' ' + str(epe_set[i]) + ' ' + str(aae_set[i]) +'\n')
    f.close()