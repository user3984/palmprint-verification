import numpy as np
import cv2 as cv
import os
import time
import math
import gc
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from ArcFace.backbone.model_mobilefacenet import MobileFaceNet
from ArcFace.backbone.model_irse import IR_50
from ArcFace.head.metrics import ArcFace
from ArcFace.util.utils import l2_norm

def getYOLOOutput(img, net):
    blobImg = cv.dnn.blobFromImage(img, 1.0/255.0, (416, 416), None, True, False) 
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

    dfg = [] #Double-finger-gap
    pc = []  #Palm-center

    if len(idxs) > 0:
        for i in idxs.flatten(): 
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            if classIDs[i]==0: dfg.append([x+w/2,y+h/2,confidences[i]])
            else: pc.append([x+w/2,y+h/2,confidences[i]])
    return dfg, pc

def onePoint(x, y, angle):
    X = x*math.cos(angle) + y*math.sin(angle)
    Y = y*math.cos(angle) - x*math.sin(angle)
    return [int(X), int(Y)]

def extractROI(img, dfg, pc):
    (H, W) = img.shape[:2]
    if W>H:
        im = np.zeros((W,W,3), np.uint8)
        im[...]=255
        im[1:H,1:W,:] = img[1:H,1:W,:]
        edge = W
    else:
        im = np.zeros((H,H,3), np.uint8)
        im[...]=255
        im[1:H,1:W,:] = img[1:H,1:W,:]
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

    k1 = (y1-y2)/(x1-x2) # line AB
    b1 = y1-k1*x1

    k2 = (-1)/k1
    b2 = y3-k2*x3

    tmpX = (b2-b1)/(k1-k2)
    tmpY = k1*tmpX+b1

    vec = [x3-tmpX, y3-tmpY]
    sidLen = math.sqrt(np.square(vec[0])+np.square(vec[1]))
    vec = [vec[0]/sidLen, vec[1]/sidLen]
    #print(vec)

    if vec[1]<0 and vec[0]>0: angle=math.pi/2-math.acos(vec[0])
    elif vec[1]<0 and vec[0]<0: angle=math.acos(-vec[0])-math.pi/2
    elif vec[1]>=0 and vec[0]>0: angle=math.acos(vec[0])-math.pi/2
    else: angle = math.pi/2-math.acos(-vec[0])
    #print(angle/math.pi*18)

    x0, y0 = onePoint(x0-edge/2, y0-edge/2, angle)

    x0 += edge/2
    y0 += edge/2

    M = cv.getRotationMatrix2D(center, angle/math.pi*180, 1.0)
    tmp = cv.warpAffine(im, M, (edge, edge))
    ROI = tmp[int(y0+unitLen/2):int(y0+unitLen*3), int(x0-unitLen*5/4):int(x0+unitLen*5/4),:]
    ROI = cv.resize(ROI, (224, 224), interpolation=cv.INTER_CUBIC)
    return ROI

def extract_features_with_uncertainty(multi_gpu, device, embedding_size, batch_size, backbone, data_loader):
    backbone.eval()
    idx = 0
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(iter(data_loader)):
            #feature, conv_feature = backbone(batch.to(device)) #IR feature, Mobile conv_feature
            feature = backbone(batch.to(device)) #IR_50
            emb_batch = feature.cpu()
            embedding = l2_norm(emb_batch)
            embeddings.append(embedding)
    embeddings = np.concatenate(embeddings, axis=0)
    print("Embeddings shape: {}".format(embeddings.shape))

    return embeddings



if __name__ == '__main__':

    '''
    Step 1: Get doble-finger-gaps and palm-center from tiny-YOLOV3-based detecor
    '''
    yolo_dir = 'path/to/yolo'
    weightsPath = os.path.join(yolo_dir, 'xxx.weights')  # YOLO weights
    configPath = os.path.join(yolo_dir, 'xxx.cfg')  # YOLO config
    imgPath1 = 'path/to/image1'  # Input images
    imgPath2 = 'path/to/image2'
    CONFIDENCE = 0.5  
    THRESHOLD = 0.4  

    net = cv.dnn.readNetFromDarknet(configPath, weightsPath)  
    print("[INFO] loading YOLO from disk...")  


    img1 = cv.imread(imgPath1)
    img2 = cv.imread(imgPath2)

    dfg1, pc1 = getYOLOOutput(img1, net)
    dfg2, pc2 = getYOLOOutput(img2, net)
    gc.collect()

    '''
    Step 2: Construct the local coordinates based on detected points.
    '''

    if len(dfg1)<2 or len(dfg2)<2: print('Detect fail. Please re-take photo and input it.')
    else:
        if len(dfg1)>2:
            tmpdfg = []
            maxD = 0
            for i in range(len(dfg1)-1):
                for j in range(i+1, len(dfg1)):
                    d = sqrt(pow(dfg1[i][0]-dfg1[j][0],2)+pow(dfg1[i][1]-dfg1[j][1],2))
                    if d>maxD:
                        tmpdfg = [dfg1[i], dfg1[j]]
                        maxD = d
            dfg1 = tmpdfg
        if len(dfg2)>2:
            tmpdfg = []
            maxD = 0
            for i in range(len(dfg2)-1):
                for j in range(i+1, len(dfg2)):
                    d = math.sqrt(pow(dfg2[i][0]-dfg2[j][0],2)+pow(dfg2[i][1]-dfg2[j][1],2))
                    if d>maxD:
                        tmpdfg = [dfg2[i], dfg2[j]]
                        maxD = d
            dfg2 = tmpdfg
        pc1 = sorted(pc1, key=lambda x:x[-1], reverse=True)
        pc2 = sorted(pc2, key=lambda x:x[-1], reverse=True)
    
        ROI1 = extractROI(img1, dfg1, pc1)
        ROI2 = extractROI(img2, dfg2, pc2)

    '''
    Step 3: Extract features from palmprint ROI and match them
    '''
    # config
    RGB_MEAN = [0.5, 0.5, 0.5] # for normalize inputs to [-1, 1]
    RGB_STD = [0.5, 0.5, 0.5]
    BACKBONE_RESUME_ROOT = '/path/to/backbone.pth'
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MULTI_GPU = True
    GPU_ID = [0]
    EMBEDDING_SIZE = 512 # feature dimension
    BATCH_SIZE = 2
    T = 0.5014

    test_transform = transforms.Compose([ 
        transforms.Resize([224, 224], interpolation= Image.NEAREST), # smaller side resized
        transforms.ToTensor(),
        transforms.Normalize(mean = RGB_MEAN, std = RGB_STD),])

    ROI1 = Image.fromarray(ROI1)
    ROI2 = Image.fromarray(ROI2)

    ROI1 = test_transform(ROI1)
    ROI2 = test_transform(ROI2)
    dataset_test = [ROI1, ROI2]

    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size = BATCH_SIZE, pin_memory = True, num_workers = 40, shuffle=False)
    #NUM_CLASS = len(test_loader.dataset.classes)
    #print("Number of Training Classes: {}".format(NUM_CLASS))
    BACKBONE = MobileFaceNet([224, 224], 3)
    if BACKBONE_RESUME_ROOT and os.path.isfile(BACKBONE_RESUME_ROOT): BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
    if MULTI_GPU:
        # multi-GPU setting
        BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
        BACKBONE = BACKBONE.to(DEVICE)
    else: BACKBONE = BACKBONE.to(DEVICE)
    embeddings = extract_features_with_uncertainty(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, test_loader)
   
    sim = np.dot(embeddings[0,:], embeddings[1,:])
    if sim>=T: print('Palmprint Verification Success.')
    else: print('Palmprint Verification Fail.')
