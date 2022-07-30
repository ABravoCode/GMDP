import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
 
EPOCHS = 16
BATCH_SIZE = 512


class MNIST_Net(torch.nn.Module):
    def __init__(self):
        super(MNIST_Net,self).__init__()     
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),

            torch.nn.Flatten(),
            torch.nn.Linear(in_features=7 * 7 * 64, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=128, out_features=10),
            torch.nn.Softmax(dim=1)
        )
    
    def forward(self,x):
        output = self.model(x)                          
        return output


def train_model():
    net = MNIST_Net()

    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize(mean=[0.5], std=[0.5])])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    path = './datasets/'
    # Setting Loss Function and Optimer with Learning rate
    lossF = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.09)

    trainData = torchvision.datasets.MNIST(path, train=True, transform=transform, download=True)
    testData = torchvision.datasets.MNIST(path, train=False, transform=transform)

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

    torch.save(net, './mnist_model.pth')
    return net

if __name__ == '__main__':
    train_model()