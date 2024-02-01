import numpy as np
import cvxpy as cp
import torchvision
from piq import haarpsi, CLIPIQA
from iqa_funs.brisque_torch import brisque
from pytorch_msssim import SSIM, MS_SSIM
from pytorch_msssim.ssim import _ssim, _fspecial_gauss_1d
import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F

ssim_module = SSIM(data_range=2, size_average=True, channel=1, win_sigma=3) # channel=1 for grayscale images
ms_ssim_module = MS_SSIM(data_range=2, size_average=True, channel=1, win_sigma=3)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_obj = lpips.LPIPS(net='alex').to(device)


def compute_LPIPS_gs(img1, img2):
    # convert to torch tensors from 2d images
    #img1 = preprocess(img1).to(device)
    #img2 = preprocess(img2).to(device)
    img1 = img1.to(device)
    img2 = img2.to(device)
    dist = lpips_obj.forward(img1, img2)
    return dist


def decompose_Q(Q):
    N = int(np.sqrt(Q.shape[0]))
    I_N2 = np.eye(N**2)
    I_N = np.eye(N)
    Q1 = cp.Variable((N, N))
    Q2 = cp.Variable((N, N))
    constrains1 = [Q1 >> 0, Q1 == Q1.T]
    constrains2 = [Q2 >> 0, Q2 == Q2.T]
    obj = cp.Minimize(cp.norm(I_N2 - cp.kron(Q1, I_N) @ Q, 'fro'))
    prob = cp.Problem(obj, constrains1)
    prob.solve()
    Q1 = Q1.value
    obj = cp.Minimize(cp.norm(I_N2 - cp.kron(Q1, Q2) @ Q, 'fro'))
    prob = cp.Problem(obj, constrains2)
    prob.solve()
    Q2 = Q2.value
    Q1 = np.linalg.inv(np.array(Q1))
    Q2 = np.linalg.inv(np.array(Q2))
    return Q1, Q2


def preprocess(img, win_size=9):
    pad_int = (win_size-1)
    img = torchvision.transforms.Pad(padding=pad_int, padding_mode='symmetric')(img)
    return img


def ssim_func(img1, img2):
    pad = (11-1)//2+1
    img1 = preprocess(img1, pad)
    img2 = preprocess(img2, pad)
    return 1-ssim_module(img1, img2)


def ms_ssim_func(img1, img2):
    pool_filt = torch.nn.functional.avg_pool2d
    levels = 5
    weights = torch.tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333]).to(img1.device)
    msssim = []
    win_sigma = 3
    win_size = 11
    win = _fspecial_gauss_1d(win_size, win_sigma)
    win = win.repeat([img1.shape[1]] + [1] * (len(img1.shape) - 1))   
    pad = (win_size-1)//2+1
    for i in range(levels):
        img1_1 = preprocess(img1, pad)
        img2_1 = preprocess(img2, pad)
        ssim_per_channel, cs = _ssim(img1_1, img2_1, data_range=2, size_average=False, win=win, K=(0.01, 0.03))
        if i < levels - 1:
            msssim.append(cs)
            padding = [s % 2 for s in img1.shape[2:]]        
            img1 = pool_filt(img1, kernel_size=2, padding=padding)
            img2 = pool_filt(img2, kernel_size=2, padding=padding)

    ssim_per_channel = torch.relu(ssim_per_channel)  # type: ignore  # (batch, channel)
    mcs_and_ssim = torch.stack(msssim + [ssim_per_channel], dim=0)  # (level, batch, channel)
    ms_ssim_val = torch.prod(mcs_and_ssim ** weights.view(-1, 1, 1), dim=0)
    return 1-ms_ssim_val.mean()


def haarpsi_func(img1, img2):
    img1 = preprocess(img1)
    img2 = preprocess(img2)
    return haarpsi(img1, img2, data_range=2)

def brisque_func(img1, img2):
    kernel_size = 7
    pad = (kernel_size-1)//2+1
    img1 = preprocess(img1, pad)
    img2 = preprocess(img2, pad)
    bris_1 = brisque(img1, data_range=2, kernel_size=kernel_size)
    bris_2 = brisque(img2, data_range=2, kernel_size=kernel_size)
    diff = (bris_2 - bris_1) - 20
    term_2 = 1/2 * (diff + torch.sqrt(torch.square(diff) + 1e-4))
    #term_2 = (bris_2 - bris_1)**2
   # term_2 = torch.maximum(bris_2 - bris_1, torch.tensor(0))
    return term_2

def clip_func(img1, img2):
    kernel_size = 7
    #pad = (kernel_size-1)//2+1
    #img1 = preprocess(img1, pad)
    #img2 = preprocess(img2, pad)
    clip_iqa = CLIPIQA(data_range=2)
    bris_1 = clip_iqa(img1)
    bris_2 = clip_iqa(img2)
    diff = (bris_1 - bris_2) + 0.005
    term_2 = 1/2 * (diff + torch.sqrt(torch.square(diff) + 1e-5))
    return term_2



model_conv = torchvision.models.resnet18(weights='IMAGENET1K_V1')
model_conv = torch.nn.Sequential(*(list(model_conv.children())[:-1]))
#model_conv = torch.load('sketch/models/trained_insects.pth')

def resnet_loss(img1, img2):

    embedding = torch.nn.Sequential(*(list(model_conv.children())[:-1])).to(device)
    embedding_2 = torch.nn.Sequential(*(list(model_conv.children())[:-2])).to(device)
    embedding_3 = torch.nn.Sequential(*(list(model_conv.children())[:-3])).to(device)
    embedding_4 = torch.nn.Sequential(*(list(model_conv.children())[:])).to(device)

    embedding.eval()
    embedding_2.eval()
    embedding_3.eval()
    embedding_4.eval()

    emb_1_a = embedding(img1)
    emb_1_b = embedding_2(img1)
    emb_1_c = embedding_3(img1)
    emb_1_d = embedding_4(img1)

    emb_2_a = embedding(img2)
    emb_2_b = embedding_2(img2)
    emb_2_c = embedding_3(img2)
    emb_2_d = embedding_4(img2)

    #with torch.no_grad():
    norm_1_a = (torch.mean(torch.square(emb_1_a), axis=1, keepdims=True))
    norm_1_b = (torch.mean(torch.square(emb_1_b), axis=1, keepdims=True))
    norm_1_c = (torch.mean(torch.square(emb_1_c), axis=1, keepdims=True))
    norm_1_d = (torch.mean(torch.square(emb_1_d), axis=1, keepdims=True))

    norm_2_a = (torch.mean(torch.square(emb_2_a), axis=1, keepdims=True))
    norm_2_b = (torch.mean(torch.square(emb_2_b), axis=1, keepdims=True))
    norm_2_c = (torch.mean(torch.square(emb_2_c), axis=1, keepdims=True))
    norm_2_d = (torch.mean(torch.square(emb_2_d), axis=1, keepdims=True))

    
    emb_1_a = emb_1_a / norm_1_a
    emb_1_b = emb_1_b / norm_1_b
    emb_1_c = emb_1_c / norm_1_c
    emb_1_d = emb_1_d / norm_1_d

    emb_2_a = emb_2_a / norm_2_a
    emb_2_b = emb_2_b / norm_2_b
    emb_2_c = emb_2_c / norm_2_c
    emb_2_d = emb_2_d / norm_2_d

    w = 0.05
    # concatenate embeddings
    #emb_1 = torch.cat(((w**3)*emb_1_a, (w**2)*emb_1_b, w*emb_1_c, emb_1_d), dim=0)
    #emb_2 = torch.cat(((w**3)*emb_2_a, (w**2)*emb_2_b, w*emb_2_c, emb_2_d), dim=0)
 
    """
    emb_1 = emb_1_d
    emb_2 = emb_2_d

    term_1 = torch.sum(torch.square(emb_1 - emb_2), axis=1)
    out = torch.mean(term_1)
    """
    
    emb_1 = emb_1_d
    emb_2 = emb_2_d

    term_1 = torch.sum(torch.square(emb_1 - emb_2), axis=1)
    out = torch.mean(term_1)
    
    return out


def resnet_loss_layer(img1, img2, layer):
    embedding = torch.nn.Sequential(*(list(model_conv.children())[:-(layer+1)])).to(device)
    embedding.eval()

    emb_1 = embedding(img1)
    emb_2 = embedding(img2)

    output = torch.mean(torch.square(emb_1 - emb_2))

    del emb_1, emb_2, embedding
    torch.cuda.empty_cache()
    return output


def vgg_loss_layer(img1, img2, layer):
    model_conv = torchvision.models.vgg16(weights='IMAGENET1K_V1')
    if (layer < 2):
        embedding = torch.nn.Sequential(*(list(model_conv.children())[:-(layer+1)]))
    else:
        embedding = torch.nn.Sequential(*(list(model_conv.children())[:-2])).to(device)
        layer -= 1
        embedding = embedding[0]
        embedding = torch.nn.Sequential(*(list(embedding.children())[:-(layer+1)])).to(device)

    embedding.eval()

    emb_1 = embedding(img1)
    emb_2 = embedding(img2)
    return torch.mean(torch.square(emb_1 - emb_2))


def alexnet_loss_layer(img1, img2, layer):
    model_conv = torchvision.models.alexnet(weights='IMAGENET1K_V1')
    if (layer == 0):
        embedding = torch.nn.Sequential(*(list(model_conv.children())[:-(layer+1)]))
    else:
        embedding = torch.nn.Sequential(*(list(model_conv.children())[:-2])).to(device)
        layer -= 1
        if (layer != 0):
            embedding = embedding[0]
            embedding = torch.nn.Sequential(*(list(embedding.children())[:-(3*layer)])).to(device)

    embedding.eval()

    emb_1 = embedding(img1)
    emb_2 = embedding(img2)
    return torch.mean(torch.square(emb_1 - emb_2))


def jacobian_loss(img1):
    #embedding = torch.nn.Sequential(*(list(model_conv.children())[:-1])).to(device)
    #embedding_2 = torch.nn.Sequential(*(list(model_conv.children())[:-2])).to(device)
    #embedding_3 = torch.nn.Sequential(*(list(model_conv.children())[:-3])).to(device)
    embedding_4 = torch.nn.Sequential(*(list(model_conv.children())[:])).to(device)

    #embedding.eval()
    #embedding_2.eval()
    #embedding_3.eval()
    embedding_4.eval()

    #emb_1_a = embedding(img1)
    #emb_1_b = embedding_2(img1)
    #emb_1_c = embedding_3(img1)
    emb_1_d = embedding_4(img1)

    #with torch.no_grad():
    #norm_1_a = (torch.mean(torch.square(emb_1_a)))
    #norm_1_b = (torch.mean(torch.square(emb_1_b)))
    #norm_1_c = (torch.mean(torch.square(emb_1_c)))
    #norm_1_d = (torch.mean(torch.square(emb_1_d)))

    #emb_1_a = emb_1_a / norm_1_a
    #emb_1_b = emb_1_b / norm_1_b
    #emb_1_c = emb_1_c / norm_1_c
    #emb_1_d = emb_1_d / norm_1_d

    #emb_1_a = emb_1_a.ravel()
    #emb_1_b = emb_1_b.ravel()
    #emb_1_c = emb_1_c.ravel()
    emb_1_d = emb_1_d.ravel()

    #w = 0.05
    # concatenate embeddings
    #emb_1 = torch.cat(((w**3)*emb_1_a, (w**2)*emb_1_b, w*emb_1_c, emb_1_d), dim=0)
    #emb_2 = torch.cat(((w**3)*emb_2_a, (w**2)*emb_2_b, w*emb_2_c, emb_2_d), dim=0)
    emb_1 = emb_1_d
    return emb_1


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    
    def fw_layer_4(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        return x
    
    def fw_layer_3(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

    def fw_layer_2(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        return x
    
    def fw_layer_1(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x
    
    def fw_layer_0(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def mnist_loss_layer(img1, img2, layer):
    model_conv = Net()
    model_conv.load_state_dict(torch.load("mnist_cnn_1.pt"))
    model_conv.eval().to(device)

    if (layer == 0):
        embedding = lambda x: model_conv.fw_layer_0(x)
    elif (layer == 1):
        embedding = lambda x: model_conv.fw_layer_1(x)
    elif (layer == 2):
        embedding = lambda x: model_conv.fw_layer_2(x)
    elif (layer == 3):
        embedding = lambda x: model_conv.fw_layer_3(x)
    else:
        embedding = lambda x: model_conv.fw_layer_4(x)


    emb_1 = embedding(img1)
    emb_2 = embedding(img2)
    return torch.mean(torch.square(emb_1 - emb_2))


