import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
 
EPOCHS = 16 
BATCH_SIZE = 8
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class CIFAR_Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()     
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 15, 3),  
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(15, 75,4),    
            torch.nn.MaxPool2d(2, 2),     
            torch.nn.Conv2d(75,375,3),   
            torch.nn.MaxPool2d(2, 2),  
            torch.nn.Flatten(),     
            torch.nn.Linear(1500,400), 
            torch.nn.ReLU(),
            torch.nn.Linear(400,120), 
            torch.nn.ReLU(),
            torch.nn.Linear(120, 84),
            torch.nn.ReLU(),
            torch.nn.Linear(84, 10)
        )
    
    def forward(self,x):
        output = self.model(x)                          
        return output


def train_model():
    net = CIFAR_Net()

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    path = './datasets/'
    # Setting Loss Function and Optimer with Learning rate
    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.002, momentum=0.9)

    trainData = torchvision.datasets.CIFAR10(path, train=True, transform=transform, download=True)
    testData = torchvision.datasets.CIFAR10(path, train=False, transform=transform)

    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE)
    testDataLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE)

    history = {'Test Loss': [], 'Test Accuracy': []}

    for epoch in range(1, EPOCHS + 1):
        processBar = tqdm(trainDataLoader, unit='step')
        net.train(True)
        for step, (trainImgs, labels) in enumerate(processBar):
            trainImgs = trainImgs.to(device)
            labels = labels.to(device)
            net = net.cuda() if torch.cuda.is_available() else net

            net.zero_grad()
            outputs = net(trainImgs)
            loss = lossF(outputs, labels)
            predictions = torch.argmax(outputs, dim=1)
            accuracy = torch.sum(predictions == labels) / labels.shape[0]

            loss.backward()
            optimizer.step()
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                       (epoch, EPOCHS, loss.item(), accuracy.item()))

            if step == len(processBar) - 1:
                correct, totalLoss = 0, 0
                net.train(False)
                with torch.no_grad():
                    for testImgs, labels in testDataLoader:
                        testImgs = testImgs.to(device)
                        labels = labels.to(device)
                        outputs = net(testImgs)
                        loss = lossF(outputs, labels)
                        predictions = torch.argmax(outputs, dim=1)

                        totalLoss += loss
                        correct += torch.sum(predictions == labels)

                        testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
                        testLoss = totalLoss / len(testDataLoader)
                    history['Test Loss'].append(testLoss.item())
                    history['Test Accuracy'].append(testAccuracy.item())

                processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                           (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(),
                                            testAccuracy.item()))
        processBar.close()

    torch.save(net, './cifar_model.pth')
    return net

if __name__ == '__main__':
    train_model()