import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from .init_cifar import CIFAR_Net
from .init_mnist import MNIST_Net

path = './datasets'
BATCH_SIZE = 1

def torchgrad(orig, labels):
    d = 32*32*3
    delta_adv = np.zeros((1,d))

    lossF = torch.nn.CrossEntropyLoss()

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    model = torch.load(r"CIFAR10_['ResNet18']_conservative_clean_model.pth")
    trainData = torchvision.datasets.CIFAR10(path, train=True, transform=transform, download=True)
    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE)
    
    orig = orig.to(device)
    labels = labels.to(device)
    orig_img = torch.tensor(orig, requires_grad=True)
    x = torch.tensor(np.clip(orig.resize(1, d).cpu().numpy()+delta_adv, -0.5, 0.5))
    outputs = model._forward_impl(orig_img)
    loss = lossF(outputs, labels)
    loss.sum().backward()

    return orig_img.grad


if __name__ == '__main__':
    gradient = torchgrad(8745)
    print(gradient)
