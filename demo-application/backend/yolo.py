import time
import math
import functools

import numpy as np
import cv2 as cv

WEIGHTSPATH = "models/yolov4-tiny-obj_best.weights"
CFGPATH = "models/yolov4-tiny-obj.cfg"
CONFIDENCE = 0.5
THRESHOLD = 0.4


def getYOLOOutput(img, net):
    blobImg = cv.dnn.blobFromImage(
        img, 1.0/255.0, (416, 416), None, True, False)
    net.setInput(blobImg)
    outInfo = net.getUnconnectedOutLayersNames()
    start = time.time()
    layerOutputs = net.forward(outInfo)
    end = time.time()
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    (H, W) = img.shape[:2]
    boxes = []
    confidences = []
    classIDs = []

    for out in layerOutputs:
        for detection in out:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)

    dfg = []  # Double-finger-gap
    pc = []  # Palm-center

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if classIDs[i] == 0:
                dfg.append([x+w/2, y+h/2, confidences[i]])
            else:
                pc.append([x+w/2, y+h/2, confidences[i]])
    return dfg, pc


def extractROI(img, dfg, pc):
    (H, W) = img.shape[:2]
    if W > H:
        im = np.zeros((W, W, 3), np.uint8)
        im[...] = 255
        im[1:H, 1:W, :] = img[1:H, 1:W, :]
        edge = W
    else:
        im = np.zeros((H, H, 3), np.uint8)
        im[...] = 255
        im[1:H, 1:W, :] = img[1:H, 1:W, :]
        edge = H

    center = (edge/2, edge/2)

    x1 = float(dfg[0][0])
    y1 = float(dfg[0][1])
    x2 = float(dfg[1][0])
    y2 = float(dfg[1][1])
    x3 = float(pc[0][0])
    y3 = float(pc[0][1])

    x0 = (x1+x2)/2
    y0 = (y1+y2)/2

    unitLen = math.sqrt(np.square(x2-x1)+np.square(y2-y1))

    k1 = (y1-y2)/(x1-x2)  # line AB
    b1 = y1-k1*x1

    k2 = (-1)/k1
    b2 = y3-k2*x3

    tmpX = (b2-b1)/(k1-k2)
    tmpY = k1*tmpX+b1

    vec = [x3-tmpX, y3-tmpY]
    sidLen = math.sqrt(np.square(vec[0])+np.square(vec[1]))
    vec = [vec[0]/sidLen, vec[1]/sidLen]
    # print(vec)

    if vec[1] < 0 and vec[0] > 0:
        angle = math.pi/2-math.acos(vec[0])
    elif vec[1] < 0 and vec[0] < 0:
        angle = math.acos(-vec[0])-math.pi/2
    elif vec[1] >= 0 and vec[0] > 0:
        angle = math.acos(vec[0])-math.pi/2
    else:
        angle = math.pi/2-math.acos(-vec[0])
    # print(angle/math.pi*18)

    x0, y0 = onePoint(x0-edge/2, y0-edge/2, angle)

    x0 += edge/2
    y0 += edge/2

    M = cv.getRotationMatrix2D(center, angle/math.pi*180, 1.0)
    tmp = cv.warpAffine(im, M, (edge, edge))
    ROI = tmp[int(y0+unitLen/2):int(y0+unitLen*3),
              int(x0-unitLen*5/4):int(x0+unitLen*5/4), :]
    ROI = cv.resize(ROI, (224, 224), interpolation=cv.INTER_CUBIC)
    return ROI


def onePoint(x, y, angle):
    X = x*math.cos(angle) + y*math.sin(angle)
    Y = y*math.cos(angle) - x*math.sin(angle)
    return [int(X), int(Y)]


@functools.lru_cache(maxsize=1)
def getYoloNet():
    return cv.dnn.readNetFromDarknet(CFGPATH, WEIGHTSPATH)
