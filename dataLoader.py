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
        if file.endswith('.jpg'):
            path = os.getcwd()
            path = path + "/datatext/" + file
            image = pil_loader(path)

            image = image.resize((150, 80), Image.ANTIALIAS)
            images.append(image)
            label = file.split(".")[0]
            label2 = []
            for i, j in enumerate(label):
                if j.isdigit():
                    j = int(j)
                    a = np.eye(62)[j]
                    label2.append(a)
                    continue
                if j.isupper():
                    j = ord(j) - ord('A') + 10
                    a = np.eye(62)[j]
                    label2.append(a)
                    continue
                if j.islower():
                    j = ord(j) - ord('a') + 36
                    a = np.eye(62)[j]
                    label2.append(a)
            i = i + 1
            if i == clip:
                break
            labels.append(label2)

    return (images, labels)