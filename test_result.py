import os.path

import torch
from torchvision import transforms

import cv2
# from PIL import Image
import numpy as np

from forest.victims.mygrad.init_resnet import ResNet18, BasicBlock

LABEL = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

device = "cuda:0" if torch.cuda.is_available() else "cpu"

if __name__ == '__main__':
    # Normalization
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    assert os.path.exists(r"forest/victims/mygrad/resnet18.pth")
    net = torch.load(r"forest/victims/mygrad/resnet18.pth", map_location=torch.device('cpu'))
    print('Success loading the model.')

    # net = net.to(device)
    path = './poisons/13Aug/targets/airplane'

    fileList = os.listdir(path)
    for f in fileList:
        pic = cv2.imread(path + '/' + f)
        img = np.array([pic])
        img = torch.from_numpy(img).float().unsqueeze(0)
        outputs = net.forward(img)
        print(outputs)