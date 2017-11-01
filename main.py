from __future__ import division, print_function, unicode_literals
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision
import torchvision.datasets as dset
from torch.autograd import Variable
import torchvision.models as models
import pickle as pkl
#%matplotlib inline
#import matplotlib.pyplot as plt
import os
import os.path
from PIL import Image
from dataLoader import importData


batchSize = 64

class CNNNetwork(torch.nn.Module):
    def __init__(self):
        super(CNNNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size = (5,5), padding = 2, stride = 2)
        # self.relu1 = nn.functional.relu(inplace = True)
        self.dropout1 = nn.Dropout(p=0.2)
        self.maxpool1 =  nn.MaxPool2d(kernel_size = (2,2), stride = 2)
        self.conv2 = nn.Conv2d(48, 64, kernel_size = (5,5), padding = 2)
        # self.relu2 = nn.functional.relu(inplace = True)
        self.dropout2 = nn.Dropout(p=0.2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2,2), stride = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = (5,5), padding = 2)
        # self.relu3 = nn.functional.relu(inplace = True)
        self.dropout3 = nn.Dropout(p=0.2)
        self.maxpool3 = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
        self.fc1 = nn.Linear(3072, 372)
        # self.relu4 = nn.functional.relu(inplace = True)
        self.dropout4 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(372, 1)

    def forward(self, x):
        out = self.conv1(x)
        out = nn.functional.relu(out)
        out = self.dropout1(out)
        out = self.maxpool1(out)

        out = self.conv2(out)
        # out = self.relu2(out)
        out = nn.functional.relu(out)
        out = self.dropout2(out)
        out = self.maxpool2(out)

        out = self.conv3(out)
        # out = self.relu3(out)
        out = nn.functional.relu(out)
        out = self.dropout3(out)
        out = self.maxpool3(out)

        out = self.fc1(out)
        # out = self.relu4(out)
        out = nn.functional.relu(out)
        out = self.dropout4(out)
        out = self.fc2(out)
        return out

model1 = CNNNetwork()
print("model:",model1)
images, labels = importData(folder = "./datatext/", clip = 10) # default directory is "./datatext/". set clip = -1 for accessing whole db 
# print(images, labels)
criterion = nn.MultiLabelSoftMarginLoss()

preprocess = transforms.Compose([

   transforms.ToTensor()
])
def train(images, labels, num_epochs = 10):
    losses = []
    for n in range(num_epochs):
        rloss = 0
        for i, image in enumerate(images):
            #image = image.cuda(async=True)
            print(i)
            image = preprocess(image)
            image = Variable(image)
            print(image)
            label = labels[i]
            label = np.asarray(label)
            label = torch.Tensor(label)
            # label = label.cuda(async=True)
            label = Variable(label)
            #print("images", images.size())
            #print("labels", labels.size())
            image = image.unsqueeze(0)
            pred_out = model1(image)
            loss = criterion(pred_out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # rloss += loss.data[0]
            losses.append(loss.data.mean())
            print("epoch:%d iteration:%d loss:%f"%(n,i,loss.data.mean()))
        # losses.append(rloss)
        print('[%d/%d] Loss: %.3f' % (epoch+1, num_epochs, np.mean(losses)))
    return losses




train(images, labels, 1)