import numpy as np
import cv2
import math
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

def EPE(ground_flow,predict_flow):
    rt = 0
    for i in range(len(ground_flow)):
        for j in range(len(ground_flow[i])):
            first = ground_flow[i][j][0]
            second = ground_flow[i][j][1]
            if first >= 2729363022:
                first = 0
            if second >= 2729363022:
                second = 0
            temp = math.pow(first - predict_flow[i][j][0],2)
            temp += math.pow(second - predict_flow[i][j][1],2)
            rt += math.sqrt(temp)

    x,y = len(ground_flow),len(ground_flow[0])
    accuracy = rt/(x*y)
    return accuracy

def AAE(ground_flow,predict_flow):
    rt = 0
    for i in range(len(ground_flow)):
        for j in range(len(ground_flow[i])):
            first = ground_flow[i][j][0]
            second = ground_flow[i][j][1]
            if first >= 2729363022:
                first = 0
            if second >= 2729363022:
                second = 0
                
            malx = math.sqrt(math.pow(first,2)+math.pow(second,2))
            maly = math.sqrt(math.pow(predict_flow[i][j][0],2)+math.pow(predict_flow[i][j][1],2))
            dotProduct = first * predict_flow[i][j][0] + second * predict_flow[i][j][1]
##            if dotProduct == None:
##                print("dotProduct is none")
##            if malx == 0 or maly == 0:
##                num = 0
##            if malx !=0 and maly != 0:
##                try:
##                    num = math.acos(dotProduct / (malx * maly))
##                except:
##                    #print(dotProduct)
##                    #print(malx,maly)
##                    print(dotProduct / (malx * maly))
            cos = float(ground_flow[i][j] * predict_flow[i][j]) / np.linalg.norm(ground_flow[i][j]) / np.linalg.norm(predict_flow[i][j])
            angle = np.arccos(cos)
            rt += angle

    x,y = len(ground_flow),len(ground_flow[0])
    accuracy = rt/(x*y)
    return accuracy

predict_flows = []
ground_truth = []

for flow in os.listdir("./middleburryflow/"):
    predict_flows.append("./middleburryflow/"+ str(flow))
for gt in os.listdir("./middleburry/"):
    ground_truth.append("./middleburry/"+str(gt))

predict_flows.sort()
ground_truth.sort()

totalEPE = 0
totalAAE = 0
for i in range(len(predict_flows)):
    ground_flow = read_flow(ground_truth[i])
    predict_flow = read_flow(predict_flows[i])
    totalEPE += EPE(ground_flow,predict_flow)
    totalAAE += abs(AAE(ground_flow,predict_flow))
    print(EPE(ground_flow,predict_flow))
    print(abs(AAE(ground_flow,predict_flow)))
    
print("Average EPE value:" + str(totalEPE/len(predict_flows)))
print("Average AAE value:" + str(totalAAE/len(predict_flows)))
