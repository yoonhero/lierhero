from re import M
import torch
import torch.nn as nn
from fastai.vision.all import *


# Linear Data Net for Sensor INPUT
class LinearNet(nn.Module):
    def __init__(self, input_size):
        super(LinearNet, self).__init__()

        self.input_size = input_size

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.input_size, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 2),

        )

    def forward(self, x):
        out = self.linear_relu_stack(x)

        return out

# Previous Multi Data Net 
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        
        self.image_features_ = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
            nn.Conv2d(16, 64, kernel_size=5, padding=2),
            nn.ReLu(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Dropout(),
        )

        # input 10 arduino sensor values
        self.numeric_features_ = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLu(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 64*64),
            nn.ReLu(inplace=True),
            nn.Dropout(),
        )

        self.combined_featuers_ = nn.Sequential(
            nn.Linear(64*64*2, 64*64*2*2),
            nn.ReLu(inplace=True),
            nn.Dropout(),
            nn.Linear(64*64*2*2, 64*64*2),
            nn.ReLu(inplace=True),
            nn.Linear(64*3*3*2, 64),
            nn.Linear(64, 2),
        )


    def forward(self, x, y):
        x = self.image_features_(x)
        x = x.view(-1, 64*64)
        y = self.numeric_features_(y)
        z = torch.cat((x, y), 1)
        z = self.combined_featuers_(z)
        return z



