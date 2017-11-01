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
    i = 0
    for file in os.listdir(folder):
        print(file)
        if file.endswith('.jpg'):
            path = os.getcwd()
            path = path + "/datatext/" + file
            image = pil_loader(path)
            image = image.resize((180, 50), Image.ANTIALIAS)
            images.append(image)
            label = file.split(".")[0]
            label2 = []
            for i, j in enumerate(label):
                if j.isupper():
                    j1 = ord(j) - ord('A') + 10
                    a = np.eye(62)[j1]
                    label2.append(a)
                if j.islower():
                    j2 = ord(j) - ord('a') + 36
                    a = np.eye(62)[j2]
                    label2.append(a)
                if j.isdigit():
                    j3 = int(j)
                    a = np.eye(62)[j3]
                    label2.append(a)
            labels.append(label2)
        i = i + 1
        if i == clip:
           break

    return (images, labels)