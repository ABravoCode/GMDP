import torch
import torchvision
import torchvision.transforms as transforms

import random
import numpy as np

from forest.victims.mygrad.delta_model import delta_model
import forest

from .init_cifar import CIFAR_Net
from .init_mnist import MNIST_Net
from .init_resnet import ResNet18
from .init_resnet import BasicBlock
from .get_autograd_in_model import torchgrad
from .delta_model import delta_model as dmod

# args = forest.options().parse_args()

SEED = 43
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True

LABEL = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def gradient_estimation_v2(mu,q,x,d,kappa,target_label,const,model,orig_img):
    sigma = 10
    f_0, ignore = function_evaluation_cons(x,kappa,target_label,const,model,orig_img)

    grad_est=0
    for i in range(q):
        u = np.random.normal(0, sigma, (1,d))
        # u = np.ones((1, d))
        u_norm = np.linalg.norm(u)
        u = u/u_norm
        f_tmp, ignore = function_evaluation_cons(x+mu*u,kappa,target_label,const,model,orig_img)
        grad_est = grad_est+ (d/q)*u*(f_tmp-f_0)/mu
    return grad_est


# f: objection function for constrained optimization formulation
def function_evaluation_cons(x, kappa, target_label, const, model, orig_img):
    # x is in [-0.5, 0.5]
    img_vec = x.clone()
    img = np.resize(img_vec, orig_img.shape)
    orig_prob, orig_class = model_prediction(model, img)
    tmp = orig_prob.clone()
    tmp[0, target_label] = 0
    tmp = tmp.detach().cpu().numpy()
    # Loss1 = const * np.max([np.log(orig_prob[0, target_label].detach().cpu().numpy() + 1e-8) - np.log(np.amax(tmp) + 1e-8), -kappa])
    Loss1 = const * np.max([(np.log(np.max(tmp) + 1e-8) - np.log(orig_prob[0, target_label].detach().cpu().numpy() + 1e-8)).all(), -kappa])
    Loss2 = np.linalg.norm(img - orig_img.detach().cpu().numpy()) ** 2 ### squared norm
    return Loss1 + Loss2, Loss2


def model_prediction(model, inputs):
    inputs = torch.from_numpy(inputs)
    prob = model.forward(inputs.cuda().float())  
    predicted_class = torch.argmax(prob)
    return prob, predicted_class


def est_grad(model, img_id):
    # config for gradient estimation
    mu = 5e-3
    q = 10
    kappa = 1e-10
    d = 32*32*3

    delta_adv = np.zeros((1,d))
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    path = './datasets/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainData = torchvision.datasets.CIFAR10(path, train=True, transform=transform, download=True)

    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=1)
    for cur_id, (trainImgs, labels) in enumerate(trainDataLoader):
        if cur_id == img_id:
            trainImgs = trainImgs.to(device)
            labels = labels.to(device)
            orig_img = trainImgs
            x = torch.tensor(np.clip(trainImgs.resize(1, d).cpu().numpy()+delta_adv, -0.5, 0.5))
            # target_label = LABEL[int(labels[0])]
            target_label = labels
            break
    
    const = 0.1
    # model = torch.load('resnet18.pth')
    model = model.to(device)
    
    d_mod, u = dmod(r"/home/mist/cloud/gmdp/CIFAR10_['ResNet18']_conservative_clean_model.pth", args)
    d_mod.to(device)
    u_norm = np.linalg.norm(u)
    u = u/u_norm
    
    f_0, ignore = function_evaluation_cons(x,kappa,target_label,const,model,orig_img)
    f_tmp, ignore = function_evaluation_cons(x,kappa,target_label,const,d_mod,orig_img)
    grad_est = u*(f_tmp-f_0)

    return grad_est


def poison_est(model, poison_img, tgt_label):
    mu = 5e-3
    q = 32
    kappa = 1e-10
    d = 500*32*32*3

    delta_adv = np.zeros((1,d))
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    orig_poison = poison_img
    x = torch.tensor(np.clip(poison_img.resize(1, d).cpu().detach().numpy()+delta_adv, -0.5, 0.5))
    target_label = tgt_label

    const = 0.1
    model = model.to(device)
    
    poison_grad_est = gradient_estimation_v2(mu,q,x,d,kappa,target_label,const,model,orig_poison)

    return torch.tensor(poison_grad_est)

'''
def est_grad(model, img_id):
    # config for gradient estimation
    mu = 5e-3
    q = 32
    kappa = 1e-10
    d = 32*32*3

    delta_adv = np.zeros((1,d))
    transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    path = './datasets/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    trainData = torchvision.datasets.CIFAR10(path, train=True, transform=transform, download=True)

    trainDataLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=1)
    for cur_id, (trainImgs, labels) in enumerate(trainDataLoader):
        if cur_id == img_id:
            trainImgs = trainImgs.to(device)
            labels = labels.to(device)
            orig_img = trainImgs
            x = torch.tensor(np.clip(trainImgs.resize(1, d).cpu().numpy()+delta_adv, -0.5, 0.5))
            # target_label = LABEL[int(labels[0])]
            target_label = labels
            break
    
    const = 0.1
    # model = torch.load('resnet18.pth')
    model = model.to(device)
    
    # -----------------------choose one--------------------------------
    # grad_est_result = torchgrad(orig_img, target_label).cpu().numpy()
    grad_est_result = gradient_estimation_v2(mu,q,x,d,kappa,target_label,const,model,orig_img)
    # -----------------------------------------------------------------

    #--------------------------------use autograd--------------------------------
    # grad_est_result = torch.autograd.grad(loss, model.parameters(), retain_graph=True, create_graph=True)
    # ----------------------------------------------------------------

    # grad_est_result = np.roll(grad_est_result, 1024)
    # grad_est_result = np.reshape(grad_est_result, (1, 3, 32, 32))
    # grad_by_torch = torchgrad(mod, img_id).cpu().numpy()  # (1, 3, 32, 32)
    
    # grad_auto = np.reshape(grad_by_torch, (1, 3072)) if mod == 'CIFAR10' else np.reshape(grad_by_torch, (1, 784))
    # grad_auto = np.roll(grad_auto, -1024)

    # Lp_Norm_2 Normalization
    # grad_est_result = grad_est_result/np.linalg.norm(grad_est_result, 2)

    # grad_auto = grad_auto/np.linalg.norm(grad_auto, 2)
    # similarity = np.vdot(grad_est_result, grad_auto)
    # sim_calculator = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
    # sim = sim_calculator(torch.tensor(grad_est_result), torch.tensor(grad_auto))
    # print(grad_est_result.shape, grad_auto.shape)
    # print('----------------------------------------------------------------------------------------')
    # print('Model:', mod)
    # print('cosine_similarity:', similarity)
    # print('torch sim:', sim)
    # np.savetxt('grad_est.txt', grad_est_result)
    # np.savetxt('grad_auto.txt', grad_auto)

    return grad_est_result
'''
