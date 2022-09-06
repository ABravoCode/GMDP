import torch

def torchgrad(orig, labels):
    d = 32*32*3

    lossF = torch.nn.CrossEntropyLoss()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(r"CIFAR10_['ResNet18']_conservative_clean_model.pth")
    
    orig = orig.to(device)
    labels = labels.to(device)
    orig_img = torch.tensor(orig, requires_grad=True)
    outputs = model._forward_impl(orig_img)

    loss = lossF(outputs, labels)
    loss.sum().backward()

    return orig_img.grad

