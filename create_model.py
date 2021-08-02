import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models


class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.num_classes = num_classes
        self.model = models.vgg16(pretrained = True)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.classifier = nn.Sequential(nn.Linear(in_features=25088, out_features=4096, bias=True),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(p=0.5, inplace=False),
                                              nn.Linear(in_features=4096, out_features=4096, bias=True),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(p=0.5, inplace=False),
                                              nn.Linear(in_features=4096, out_features=self.num_classes, bias=True),
                                              nn.Sigmoid())
    def forward(self, x):
        x = self.model(x)
        return x
