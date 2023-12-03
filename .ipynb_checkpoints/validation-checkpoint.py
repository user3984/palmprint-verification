
import torch
from mobilevit import MobileViT


model = MobileViT(arch='xx_small')
state_dict = torch.load('/root/palmprint/model_weights_best.pth')
model.load_state_dict(state_dict)

with torch.no_grad():
  features = model(roi)
  features = F.adaptive_avg_pool2d(features, (1, 1)).flatten(1)