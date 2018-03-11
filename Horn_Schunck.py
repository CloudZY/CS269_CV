import cv
import numpy as np
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

if __name__ == '__main__':
    prev_gray = cv.LoadImage('./other-data-gray/Beanbags/frame10.png', cv.CV_LOAD_IMAGE_GRAYSCALE)
    curr_gray = cv.LoadImage('./other-data-gray/Beanbags/frame11.png', cv.CV_LOAD_IMAGE_GRAYSCALE)

    prev_color = cv.LoadImage('./other-data-gray/Beanbags/frame10.png', cv.CV_LOAD_IMAGE_COLOR)
    curr_color = cv.LoadImage('./other-data-gray/Beanbags/frame11.png', cv.CV_LOAD_IMAGE_COLOR)

    cols = prev_gray.width
    rows = prev_gray.height

    velx = cv.CreateMat(rows, cols, cv.CV_32FC1)
    vely = cv.CreateMat(rows, cols, cv.CV_32FC1)
    cv.SetZero(velx)
    cv.SetZero(vely)

    cv.CalcOpticalFlowHS(prev_gray, curr_gray, False, velx, vely, 100.0,
                         (cv.CV_TERMCRIT_ITER | cv.CV_TERMCRIT_EPS, 64, 0.01))

    # cv.CalcOpticalFlowLK(prev_gray, curr_gray, (15, 15), velx, vely)

    cv.NamedWindow("Optical flow HS")
    cv.ShowImage("Optical flow HS", draw_flow(prev_color, velx, vely))
    # cv.SaveImage("resultHS.png", desImageHS)

    cv.NamedWindow("flow HSV")
    cv.ShowImage("flow HSV", draw_hsv(velx, vely))

    cv.NamedWindow("glitch")
    cv.ShowImage("glitch", warp_flow(prev_color, velx, vely))

    cv.WaitKey(0)
    cv.DestroyAllWindows()