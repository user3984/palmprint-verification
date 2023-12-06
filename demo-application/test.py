

from PIL import Image
from torchvision import transforms
from mobilevit import MobileViT
import torch
import os
import cv2 as cv
import gc
import time
import numpy as np
import math

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
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))

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


yolo_dir = 'models'
weightsPath = os.path.join(
    yolo_dir, 'yolov4-tiny-obj_best.weights')  # YOLO weights
configPath = os.path.join(yolo_dir, 'yolov4-tiny-obj.cfg')  # YOLO config
imgPath1 = 'data/1.jpg'  # Input images
imgPath2 = 'data/3.jpg'


net = cv.dnn.readNetFromDarknet(configPath, weightsPath)
print("[INFO] loading YOLO from disk...")

img1 = cv.imread(imgPath1)
img2 = cv.imread(imgPath2)

dfg1, pc1 = getYOLOOutput(img1, net)
dfg2, pc2 = getYOLOOutput(img2, net)
print(len(dfg1), len(dfg2))
gc.collect()

if len(dfg1) > 2:
    tmpdfg = []
    maxD = 0
    for i in range(len(dfg1)-1):
        for j in range(i+1, len(dfg1)):
            d = math.sqrt(pow(dfg1[i][0]-dfg1[j][0], 2) +
                          pow(dfg1[i][1]-dfg1[j][1], 2))
            if d > maxD:
                tmpdfg = [dfg1[i], dfg1[j]]
                maxD = d
    dfg1 = tmpdfg

if len(dfg2) > 2:
    tmpdfg = []
    maxD = 0
    for i in range(len(dfg2)-1):
        for j in range(i+1, len(dfg2)):
            d = math.sqrt(pow(dfg2[i][0]-dfg2[j][0], 2) +
                          pow(dfg2[i][1]-dfg2[j][1], 2))
            if d > maxD:
                tmpdfg = [dfg2[i], dfg2[j]]
                maxD = d
    dfg2 = tmpdfg

pc1 = sorted(pc1, key=lambda x: x[-1], reverse=True)
pc2 = sorted(pc2, key=lambda x: x[-1], reverse=True)


def show(img):
    cv.imshow("", img)
    cv.waitKey(0)
    cv.destroyAllWindows()


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


ROI1 = extractROI(img1, dfg1, pc1)
ROI2 = extractROI(img2, dfg2, pc2)


# """"""""""""""""""""""""""""""""""""""

# # Open a connection to the camera (0 by default represents the default camera)
# cap = cv.VideoCapture(0)

# # Check if the camera is opened successfully
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()

#     # If the frame is read correctly, ret will be True
#     if ret:
#         # Display the frame
#         dfg, pc = getYOLOOutput(frame, net)
#         gc.collect()
#         try:
#             assert len(dfg) >= 2
#             if len(dfg) > 2:
#                 tmpdfg = []
#                 maxD = 0
#                 for i in range(len(dfg)-1):
#                     for j in range(i+1, len(dfg)):
#                         d = math.sqrt(pow(dfg[i][0]-dfg[j][0], 2) +
#                                       pow(dfg[i][1]-dfg[j][1], 2))
#                         if d > maxD:
#                             tmpdfg = [dfg[i], dfg[j]]
#                             maxD = d
#                 dfg = tmpdfg
#             pc = sorted(pc, key=lambda x: x[-1], reverse=True)
#             roi = extractROI(frame, dfg, pc)
#             cv.imshow('Camera Feed', roi)
#         except:
#             cv.imshow('Camera Feed', frame)

#     # Break the loop if 'q' key is pressed
#     if cv.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the camera and close the window
# cap.release()
# cv.destroyAllWindows()
""""""""""""""""""
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
ROI1 = transform(Image.fromarray(ROI1)).unsqueeze(0)


model = MobileViT(arch='x_small', last_channels=1024, gd_conv=True)
state_dict = torch.load(
    'models/x_small_model_weights_best.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
with torch.no_grad():
    features = model(ROI1)
    norm_feat = (features**2).sum(axis=1, keepdim=True).sqrt()
    features = features / norm_feat
    print(list(features.numpy()[0]))
    # calculate similarities
    # make sure your features stored in database are normalized
