import os
import numpy as np
import cv2

def draw_flow(img, velx, vely, step=16):
    cols, rows = img.shape[:2]
    for i in range(0, cols, step):
        for j in range(0, rows, step):
            dx = int(velx[i][j])
            dy = int(vely[i][j])
            cv2.line(img, (j, i), (j + dy, i + dx), (0, 255, 0))
            cv2.circle(img, (j, i), 1, (0, 255, 0), -1)
    return img

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

predict_flows = []
images = []
for flow in os.listdir("./flow/"):
    predict_flows.append("./flow/" + str(flow))
for image in os.listdir("./images/"):
    images.append("./images/"+str(image))
print(predict_flows)

predict_flows.sort()
images.sort()

for index in range(len(predict_flows)):
    frame = cv2.imread(images[index])
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    flow = read_flow(predict_flows[index])
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

    img = draw_flow(frame,velx,vely)
    #img = cv2.add(frame,mask)
    cv2.imshow(images[0], img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()

cv2.destroyAllWindows()