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
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs
    
    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(20):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()

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
        # self.v = nn.View(14080)
        self.fc1 = nn.Linear(14080, 3072)
        # self.relu4 = nn.functional.relu(inplace = True)
        self.dropout4 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(3072,372)

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
        # out = self.v(out)
        out = out.view(14080)
        out = self.fc1(out)
        # out = self.relu4(out)
        out = nn.functional.relu(out)
        out = self.dropout4(out)
        out = self.fc2(out)
        return out

encoder = EncoderCNN(args.embed_size)
decoder = DecoderRNN(args.embed_size, args.hidden_size, 
                         len(vocab), args.num_layers)
if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()

print("model:",encoder);

model1.cuda()
model1 = torch.nn.parallel.DataParallel(model1)
images, labels, labelsNormal = importData(folder = "./datatext/", clip = -1) # default directory is "./datatext/". set clip = -1 for accessing whole db 
print("Data import completed")
# print("sdad")
# print(images, labels)
criterion = nn.MultiLabelSoftMarginLoss()

preprocess = transforms.Compose([

   transforms.ToTensor()
])
optimizer = torch.optim.Adam(model1.parameters(), lr = 0.0001)
def train(images, labels, num_epochs = 20):
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
        label1 = torch.Tensor(label1)
        # label = label.cuda()
        image1 = image1.cuda()
        image1 = Variable(image1)
        image1 = image1.unsqueeze(0)
        label1 = label1.cuda()
        label1 = Variable(label1)
        pred_out = model1(image1)
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


train(images, labels, 10)
images, labels, labelsNormal = importData(folder = "./full/", clip = 10) # default directory is "./datatext/". set clip = -1 for accessing whole db 
test(images, labels, labelsNormal)
