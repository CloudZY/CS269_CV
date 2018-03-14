import numpy as np
import cv2
import os

def read_flow(filename):
        f = open(filename, 'rb')
        magic = np.fromfile(f, np.float32, count=1)
        data2d = None

        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
        else:
            w = np.fromfile(f, np.int32, count=1)[0]
            h = np.fromfile(f, np.int32, count=1)[0]
            print("Reading %d x %d flo file" % (h, w))
            data2d = np.fromfile(f, np.float32, count=2 * w * h)
            # reshape data into 3D array (columns, rows, channels)
            data2d = np.resize(data2d, (h, w, 2))
        f.close()
        return data2d

def save_flow(flowname,imagename,outname):
    flow = read_flow(flowname)
    #flow = read_flow("./flow/sintel.flo")
    print(flowname,imagename,outname)
    velx = []
    vely = []
    for row in flow:
        newx = []
        newy = []
        for col in row:
            newx.append(col[0])
            newy.append(col[1])
        velx.append(newx)
        vely.append(newy)

    def draw_flow(img, velx, vely, step=16):
        cols,rows = img.shape[:2]
        for i in range(0, cols, step):
            for j in range(0, rows, step):
                dx = int(velx[i][j])
                dy = int(vely[i][j])
                cv2.line(img, (j, i), (j + dy, i + dx), (0, 255, 0))
                cv2.circle(img, (j, i), 1, (0, 255, 0), -1)
        return img

    curr = cv2.imread(imagename)
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    out_flow = draw_flow(curr,velx,vely)

    cv2.imwrite(outname,out_flow)

    ##def draw_flow(img, flow, step=16):
    ##    h, w = img.shape[:2]
    ##    y, x = np.mgrid[step/2 : h : step, step/2 : w : step].reshape(2, -1)
    ##    fx, fy = flow[y, x].T
    ##    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
    ##    lines = np.int32(lines + 0.5)
    ##    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    ##    cv2.polylines(vis, lines, 0, (0, 255, 0))
    ##    for (x1, y1), (x2, y2) in lines:
    ##        cv2.circle(vis, (x1, y1), 1, (0, 255, 0), -1)
    ##    return vis
    ##
    ##curr = cv2.imread("./flow/frame10.png")
    ##curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
    ##out_flow = draw_flow(curr_gray,flow)
    ##
    ##cv2.imshow('./flow/frame10.png',out_flow)

flows = []
images = []
outnames = []

for flow in os.listdir("./flow/"):
    flows.append("./flow/"+ str(flow))
for image in os.listdir("./images/"):
    images.append("./images/"+str(image))
images = images[:-1]
for i in range(len(flows)):
    outnames.append("./images_after_flow/" + "out_image_" + str(i+1)+".png")
for i in range(len(flows)):
    save_flow(flows[i],images[i],outnames[i])

