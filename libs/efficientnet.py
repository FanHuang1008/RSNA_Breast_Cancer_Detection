import numpy as np
from copy import deepcopy
from tqdm import tqdm

import torch
from torch import nn
from torchvision import models

class EffNetModel(nn.Module):
    def __init__(self):
        super(EffNetModel, self).__init__()
        backbone = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.DEFAULT)
        backbone.classifier = nn.Linear(1408, 1)
        self.backbone = backbone
        
    def forward(self, dcm):
        out = self.backbone(dcm)
        out = torch.sigmoid(out)
        return out

def predict(models, loader):
    preds = [[] for _ in range(len(models))]

    for x in tqdm(loader):
        x = x.cuda()
        with torch.no_grad():
            for i, model in enumerate(models):
                outputs = model(x)
                outputs = outputs.cpu().detach().squeeze(1).tolist()
                preds[i].extend(outputs)
        
    preds = np.mean(preds,0)

    return preds

class ModelEma(torch.nn.Module):
    def __init__(self, model, decay=0.999, device=None):
        super().__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device  # perform ema on different device from model if set
        if self.device is not None:
            self.module.to(device=device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)