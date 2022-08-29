import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


BN_MOMENTUM = 0.1


def split_branch(opt):
    return nn.Sequential(
            nn.Conv2d(opt.d_model, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, opt.d_branch, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(opt.d_branch, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


class RegressionFC(nn.Module):
    def __init__(self, opt):
        super(RegressionFC, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(opt.d_branch * 2, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
            nn.Conv2d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        )
        self.regressor = nn.Sequential(
            nn.Linear(opt.dim_fc, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128, momentum=BN_MOMENTUM),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128, momentum=BN_MOMENTUM),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        feats = self.features(x)
        feats_flat = feats.view(feats.shape[0], -1)
        out = self.regressor(feats_flat)
        return out
