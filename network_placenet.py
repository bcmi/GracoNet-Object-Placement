import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(*list(models.resnet18(pretrained=True).children())[:-1])

    def forward(self, x):
        output = self.net(x).view(x.size(0), -1)
        return output
