import torch
import torch.nn as nn
import torchvision
import dataLoader
from PIL import Image




batchSize = 64

def CNNNetwork():
    def __init__(self):
        super(CNNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size = (5,5), padding = 2, stride = 2)
        self.relu1 = nn.ReLu(inplace = True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.maxpool1 =  nn.MaxPool2d(kernel_size = (2,2), stride = 2)
        self.conv2 = nn.Conv2d(48, 64, kernel_size = (5,5), padding = 2)
        self.relu2 = nn.ReLu(inplace = True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2,2), stride = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = (5,5), padding = 2)
        self.relu3 = nn.ReLu(inplace = True)
        self.dropout3 = nn.Dropout(p=0.2)
        self.maxpool3 = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
        self.fc1 = nn.Linear(3072, 372)
        self.relu4 = nn.ReLu(inplace = True)
        self.dropout4 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(372, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.dropou1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        out = self.relu2(out)
        out = self.dropou2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        out = self.relu3(out)
        out = self.dropou3(out)
        out = self.maxpool3(out)

        out = self.fc1(out)
        out = self.relu4(out)
        out = self.dropout4(out)
        out = self.fc2(out)

2
