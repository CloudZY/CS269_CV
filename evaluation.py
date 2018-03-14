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
            if abs(first) >= 1e9 or abs(second) >= 1e9:
                first,second =0,0
            temp = math.pow(first - predict_flow[i][j][0],2)
            temp += math.pow(second - predict_flow[i][j][1],2)
            rt += math.sqrt(temp)

def EPE_RB(ground_flow,predict_flow):
    rt = 0
    for i in range(len(ground_flow)):
        for j in range(len(ground_flow[i])):
            first = -1*ground_flow[i][j][0]
            second = -1*ground_flow[i][j][1]
            if abs(first) >= 1e9 or abs(second) >= 1e9:
                first,second = 0,0
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
            if abs(first) >= 1e9 or abs(second) >= 1e9:
                continue
            malx = math.sqrt(math.pow(first,2)+math.pow(second,2))
            maly = math.sqrt(math.pow(predict_flow[i][j][0],2)+math.pow(predict_flow[i][j][1],2))
            dotProduct = first * predict_flow[i][j][0] + second * predict_flow[i][j][1]
            if dotProduct == None:
                continue
            if malx == 0 or maly == 0:
                continue
            if malx !=0 and maly != 0:
                try:
                    num = math.acos(dotProduct / (malx * maly))
                except:
                    print(first, second, predict_flow[i][j][0], predict_flow[i][j][1])
                    print(dotProduct)
                    print(malx,maly)
                    print(dotProduct / (malx * maly))
            rt += num

    x,y = len(ground_flow),len(ground_flow[0])
    accuracy = rt/(x*y)
    return accuracy

def AAE_RB(ground_flow,predict_flow):
    rt = 0
    for i in range(len(ground_flow)):
        for j in range(len(ground_flow[i])):
            first = -1*ground_flow[i][j][0]
            second = -1*ground_flow[i][j][1]
            if abs(first) >= 1e9 or abs(second) >= 1e9:
                continue
            malx = math.sqrt(math.pow(first,2)+math.pow(second,2))
            maly = math.sqrt(math.pow(predict_flow[i][j][0],2)+math.pow(predict_flow[i][j][1],2))
            dotProduct = first * predict_flow[i][j][0] + second * predict_flow[i][j][1]
            if dotProduct == None:
                continue
            if malx == 0 or maly == 0:
                continue
            if malx !=0 and maly != 0:
                try:
                    num = math.acos(dotProduct / (malx * maly))
                except:
                    print(first, second, predict_flow[i][j][0], predict_flow[i][j][1])
                    print(dotProduct)
                    print(malx,maly)
                    print(dotProduct / (malx * maly))
            rt += num

    x,y = len(ground_flow),len(ground_flow[0])
    accuracy = rt/(x*y)
    return accuracy

def run_RB():
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
        totalEPE += EPE_RB(ground_flow,predict_flow)
        totalAAE += AAE_RB(ground_flow,predict_flow)

    print("Average EPE value:" + str(totalEPE/len(predict_flows)))
    print("Average AAE value:" + str(totalAAE/len(predict_flows)))

def run():
    predict_flows = []
    ground_truth = []

    for flow in os.listdir("./flow/"):
        predict_flows.append("./flow/"+ str(flow))
    for gt in os.listdir("./ground_truth/"):
        ground_truth.append("./ground_truth/"+str(gt))

    predict_flows.sort()
    ground_truth.sort()

    totalEPE = 0
    totalAAE = 0
    for i in range(len(predict_flows)):
        ground_flow = read_flow(ground_truth[i])
        predict_flow = read_flow(predict_flows[i])
        totalEPE += EPE(ground_flow,predict_flow)
        totalAAE += AAE(ground_flow,predict_flow)

    print("Average EPE value:" + str(totalEPE/len(predict_flows)))
    print("Average AAE value:" + str(totalAAE/len(predict_flows)))

run_RB()