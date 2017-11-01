import os
import sys
from PIL import Image
import numpy as np
from torchvision import transforms
import torch

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def importData(folder = "./datatext/", clip = -1):
    images = []
    labels = []
    labelsNormal = []
    i1 = 0
    for file in os.listdir(folder):
        # print(file)
        if file.endswith('.jpg'):
            path = os.getcwd()
            path = path + "/datatext/" + file
            image = pil_loader(path)
            image = image.resize((180, 50), Image.ANTIALIAS)
            images.append(image)
            label = file.split(".")[0]
            label2 = []
            label2Normal = []
            for i, j in enumerate(label):
                if j.isupper():
                    j1 = ord(j) - ord('A') + 10
                    a = np.eye(62)[j1]
                    label2.append(a)
                    label2Normal.append(j1)
                if j.islower():
                    j2 = ord(j) - ord('a') + 36
                    a = np.eye(62)[j2]
                    label2.append(a)
                    label2Normal.append(j2)
                if j.isdigit():
                    j3 = int(j)
                    a = np.eye(62)[j3]
                    label2.append(a)
                    label2Normal.append(j3)
            labels.append(label2)
            labelsNormal.append(label2Normal)
        # print("before",i1)
        i1 = i1 + 1
        # print("adas",i1)
        if i1 == clip:
           break

    return (images, labels, labelsNormal)