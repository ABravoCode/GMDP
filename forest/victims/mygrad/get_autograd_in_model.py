import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

from init_cifar import CIFAR_Net
from init_mnist import MNIST_Net

path = './datasets'
BATCH_SIZE = 1

def torchgrad(mod, img_id):
    if mod == 'CIFAR10':
        d = 32*32*3
        delta_adv = np.zeros((1,d))

        lossF = torch.nn.CrossEntropyLoss()

        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    

        model = torch.load('cifar_model.pth')
        trainData = torchvision.datasets.CIFAR10(path, train=True, transform=transform, download=True)
        trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE)
    
    elif mod == 'MNIST':
        d = 28*28
        delta_adv = np.zeros((1,d))

        lossF = torch.nn.CrossEntropyLoss()

        transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        model = torch.load('mnist_model.pth')
        trainData = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
        trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE)

    else:
        raise NameError('Unknown model')
    
    for cur_id, (trainImgs, labels) in enumerate(trainDataLoader):
        if cur_id == img_id:
            trainImgs = trainImgs.to(device)
            labels = labels.to(device)
            orig_img = torch.tensor(trainImgs, requires_grad=True)
            x = torch.tensor(np.clip(trainImgs.resize(1, d).cpu().numpy()+delta_adv, -0.5, 0.5))
            outputs = model.forward(orig_img)
            loss = lossF(outputs, labels)
            loss.sum().backward()
            break
    return orig_img.grad


if __name__ == '__main__':
    gradient = torchgrad('MNIST', 0)
    print(gradient)
