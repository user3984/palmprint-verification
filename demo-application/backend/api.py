import functools
import gc
import math
from typing import Optional, List

import torch
from PIL import Image
from torchvision import transforms

from backend.mobilevit import MobileViT
from .yolo import getYOLOOutput, extractROI, getYoloNet


def try_get_roi(frame) -> Optional[List]:
    net = getYoloNet()
    dfg, pc = getYOLOOutput(frame, net)
    gc.collect()
    try:
        assert len(dfg) >= 2
        if len(dfg) > 2:
            tmpdfg = []
            maxD = 0
            for i in range(len(dfg)-1):
                for j in range(i+1, len(dfg)):
                    d = math.sqrt(pow(dfg[i][0]-dfg[j][0], 2) +
                                  pow(dfg[i][1]-dfg[j][1], 2))
                    if d > maxD:
                        tmpdfg = [dfg[i], dfg[j]]
                        maxD = d
            dfg = tmpdfg
        pc = sorted(pc, key=lambda x: x[-1], reverse=True)
        roi = extractROI(frame, dfg, pc)
        return roi
    except:
        return None


def roi_to_embeddings(roi) -> List:
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    roi = transform(Image.fromarray(roi)).unsqueeze(0)
    model = _get_model()

    with torch.no_grad():
        features = model(roi)
        norm_feat = (features**2).sum(axis=1, keepdim=True).sqrt()
        features = features / norm_feat
        return list(features.numpy()[0])


@functools.lru_cache()
def _get_model():
    model = MobileViT(arch='x_small', last_channels=1024, gd_conv=True)
    state_dict = torch.load(
        'models/x_small_model_weights_best.pth', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    return model
