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
import os
import os.path
from PIL import Image
from dataLoader import importData


batchSize = 64

class CNNNetwork(torch.nn.Module):
    def __init__(self):
#	vgg = nn.Sequential()
#	add.(nn.Reshape(1,50,170))
	
        super(CNNNetwork, self).__init__()
#	self.shape = (1,50,170)
	
#        self.conv1 = nn.Conv2d(1, 48, kernel_size = (5,5), padding = 2, stride = 2)
#	MaxPooling = nn.SpatialMaxPooling
	
        self.conv1 = nn.Conv2d(1, 64, kernel_size = (5,5), stride = 2)
       # self.relu1 = nn.functional.relu(inplace = True)
        self.dropout1 = nn.Dropout(p=0.2)
#        self.maxpool1 =  nn.MaxPool2d(kernel_size = (2,2), stride = 2)
        self.conv2 = nn.Conv2d(48, 64, kernel_size = (5,5), padding = 2)
#
#        # self.conv2 = nn.Conv2d(48, 64, kernel_size = (5,5))
       # self.relu2 = nn.functional.relu(inplace = True)
        self.maxpool1 = nn.MaxPool2d(kernel_size = (2,2), stride = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = (5,5), padding = 2)
       # self.relu3 = nn.functional.relu(inplace = True)
        self.dropout2 = nn.Dropout(p=0.4)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = (5,5))
        #self.relu4 = nn.functional.relu(inplace = True)
#        self.dropout3 = nn.Dropout(p=0.2)
        self.maxpool2 = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
#        # self.v = nn.View(14080)

        self.conv5 = nn.Conv2d(128, 256, kernel_size = (5,5), padding = 2)
       # self.relu5 = nn.functional.relu(inplace = True)
        self.dropout3 = nn.Dropout(p=0.4)
        self.conv6 = nn.Conv2d(256, 256, kernel_size = (5,5))
        #self.relu6 = nn.functional.relu(inplace = True)

        self.conv6 = nn.Conv2d(256, 256, kernel_size = (5,5))
        #self.relu6 = nn.functional.relu(inplace = True)
        self.dropout3 = nn.Dropout(p=0.4)
        self.maxpool3 = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
#        self.fc1 = nn.Linear(14080, 3072)
        self.conv7 = nn.Conv2d(256, 512, kernel_size = (5,5), padding = 2)
        #self.relu7 = nn.functional.relu(inplace = True)
        self.dropout5 = nn.Dropout(p=0.4)
        self.conv8 = nn.Conv2d(512, 512, kernel_size = (5,5))
        #self.relu8 = nn.functional.relu(inplace = True)
        self.dropout6 = nn.Dropout(p=0.4)
        self.conv8 = nn.Conv2d(512, 512, kernel_size = (5,5))
        #self.relu8 = nn.functional.relu(inplace = True)
        self.maxpool4 = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
        self.conv8 = nn.Conv2d(512, 512, kernel_size = (5,5))
        #self.relu8 = nn.functional.relu(inplace = True)
        self.dropout6 = nn.Dropout(p=0.4)
        self.conv8 = nn.Conv2d(512, 512, kernel_size = (5,5))
        #self.relu8 = nn.functional.relu(inplace = True)
        self.dropout6 = nn.Dropout(p=0.4)
#        # self.relu4 = nn.functional.relu(inplace = True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size = (5,5))
#        self.relu8 = nn.functional.relu(inplace = True)
        #self.dropout6 = nn.Dropout(p=0.4)
#        self.dropout4 = nn.Dropout(p=0.2)
#        self.fc2 = nn.Linear(3072,372)
        self.maxpool4 = nn.MaxPool2d(kernel_size = (2,2), stride = 2)
#	self.view1 = nn.View(512*2*6)	
	self.cl1 = nn.Linear(512*2*6,512)
#	self.cl2 = nn.BatchNormalization(512)
	self.cl3 = nn.Linear(512,372)
 
    def forward(self, x):
	out = self.shape(x)
        out = self.conv1(out)
        out = nn.functional.relu(out)
        out = self.dropout1(out)
        out = self.conv2(out)
        out = nn.functional.relu(out)
        out = self.maxpool1(out)

        out = self.conv3(out)
         # out = self.relu2(out)
        out = nn.functional.relu(out)
        out = self.dropout2(out)
        out = self.conv3(out)
        out = nn.functional.relu(out)
        out = self.dropout2(out)
        out = self.conv4(out)
        out = nn.functional.relu(out)
        out = self.maxpool2(out)

        out = self.conv5(out)
         # out = self.relu3(out)
        out = nn.functional.relu(out)
        out = self.dropout3(out)
        out = self.conv6(out)
        out = nn.functional.relu(out)
        out = self.dropout3(out)
        out = self.maxpool3(out)

        out = self.conv7(out)
        out = nn.functional.relu(out)
        out = self.dropout3(out)

        out = self.conv8(out)
        out = nn.functional.relu(out)
        out = self.dropout3(out)

        out = self.conv8(out)
	out = nn.functional.relu(out)
        out = self.conv8(out)
        #out = nn.functional.relu(out)
 # out = self.v(out)
        out = self.maxpool4(out)
        out = out.view(512*2*6)
        out = self.cl1(out)
 #       out = self.cl2(out)
        out = self.cl3(out)
         # out = self.relu4(out)
       # out = nn.functional.relu(out)
       # out = self.dropout4(out)
       # out = self.fc2(out)
        return out





model1 = CNNNetwork()
print("model:",model1)

model1.cuda()
model1 = torch.nn.parallel.DataParallel(model1)
images, labels, labelsNormal = importData(folder = "../Data100/", clip = 50) # default directory is "./datatext/". set clip = -1 for accessing whole db 
print("Data import completed")
# print("sdad")
# print(images, labels)
criterion = nn.MultiLabelSoftMarginLoss().cuda()

preprocess = transforms.Compose([

   transforms.ToTensor()
])
optimizer = torch.optim.Adam(model1.parameters(), lr = 0.000001)
def train(model1,images, labels, num_epochs = 50):
    losses = []
    for n in range(num_epochs):
        rloss = 0
        images_ = []
        for i, image in enumerate(images):
            #image = image.cuda(async=True)
            # print(i)
            image = preprocess(image)
            image = image.cuda()
            image = Variable(image)
            # print(image)
            label = labels[i]
            label = np.asarray(label)
            label = torch.Tensor(label)
            label = label.cuda(async=True)
            label = Variable(label)
            #print("images", images.size())
            #print("labels", labels.size())
            image = image.unsqueeze(0)
            pred_out = model1(image)
            # print(label, pred_out)
            loss = criterion(pred_out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # rloss += loss.data[0]
            losses.append(loss.data.mean())
            print("epoch:%d iteration:%d loss:%f"%(n,i,loss.data.mean()))
        # losses.append(rloss)
        print('[%d/%d] Loss: %.3f' % (n+1, num_epochs, np.mean(losses)))
	 # ... after training, save your model 
        #torch.save(model1,'/users/btech/dsinghvi/mytraining.pt')
        torch.save(model1,'resnet.pt')
    return losses


def test(images, labels, labelsNormal):
    # Write loops for testing the model on the test set
    # You should also print out the accuracy of the model
    
    # .. to load your previously training model:
    #model1 = torch.load('/users/btech/dsinghvi/mytraining.pt')
    model1 = torch.load('resnet.pt')
    model1.eval()
    correct = 0
    total = 0
    
    for i, image1 in enumerate(images):
        predicted_label = []
        image1 = preprocess(image1)
        # image1 = image1.cuda()
        label1 = labels[i]
        label1 = np.asarray(label1)
        # label = label.cuda()
        label1 = torch.Tensor(label1)
        image1 = image1.cuda()
        image1 = Variable(image1)
        image1 = image1.unsqueeze(0)
        label1 = label1.cuda()
        label1 = Variable(label1)
        pred_out = model1(image1)
        print(pred_out)
        pred_out = pred_out.transpose(0,1)
        val, idx1 = torch.max(pred_out[:62], 0)
        # idx1 = idx1.numpy()
        idx1 = idx1.cpu().data.numpy()[0]
        # print(idx1)
        # predicted_label.append(np.eye(62)[idx1])
        val, idx2 = torch.max(pred_out[62:124], 0)
        
        idx2 = idx2.cpu().data.numpy()[0]
        # predicted_label.append(np.eye(62)[idx2])
        val, idx3 = torch.max(pred_out[124:186], 0)
        
        idx3 = idx3.cpu().data.numpy()[0]
        # predicted_label.append(np.eye(62)[idx3])
        val, idx4 = torch.max(pred_out[186:248], 0)
        idx4 = idx4.cpu().data.numpy()[0]

        # predicted_label.append(np.eye(62)[idx4])
        val, idx5 = torch.max(pred_out[248:310], 0)
        
        idx5 = idx5.cpu().data.numpy()[0]
        # predicted_label.append(np.eye(62)[idx5])
        val, idx6 = torch.max(pred_out[310:372], 0)
        
        idx6 = idx6.cpu().data.numpy()[0]
        # predicted_label.append(np.eye(62)[idx6])

        # predicted_label = np.asarray(predicted_label)
        # print(predicted_label)
        if idx1 == labelsNormal[i][0] and idx2 == labelsNormal[i][1] and idx3 == labelsNormal[i][2] and idx4 == labelsNormal[i][3] and idx5 == labelsNormal[i][4] and idx6 == labelsNormal[i][5]:
            correct = correct + 1
        print(idx1, labelsNormal[i][0])
        print(idx2, labelsNormal[i][1])
        print(idx3, labelsNormal[i][2])
        print(idx4, labelsNormal[i][3])
        print(idx5, labelsNormal[i][4])
        print(idx6, labelsNormal[i][5])
        # predicted_label = torch.Tensor(predicted_label)
        # predicted_label = Variable(predicted_label)
        # if(predicted_label == label1):
            # correct = correct + 1
        total += label1.size(0)
        print("total:%d correct:%d"%(total, correct))
    print("Accuracy:",(100*correct/total))

resnet18 = models.resnet18(pretrained=True)
resnet18.fc = nn.Linear(resnet18.fc.in_features, 372)
resnet18.cuda()
resnet18 = torch.nn.parallel.DataParallel(resnet18)
train(resnet18, images, labels, 10)
print("reached")
images, labels, labelsNormal = importData(folder = "./data2/", clip = 1000) # default directory is "./datatext/". set clip = -1 for accessing whole db 
test(images, labels, labelsNormal)
