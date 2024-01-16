from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from compute_q.compute_Q_dct_pp import compute_Q_layer_mnist_dct as compute_Q
import matplotlib.pyplot as plt
import numpy as np
from utils.utils_lpit import dct_2d
# Training settings

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

use_cuda = torch.cuda.is_available()
true_N = (28, 28)

compute_Q_obj = compute_Q(true_N=true_N, n_cwd=2, N=4)


torch.manual_seed(0)

if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

test_kwargs = {'batch_size': 1}

if use_cuda:
    cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
    test_kwargs.update(cuda_kwargs)

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
    ])

dataset2 = datasets.MNIST('../data', train=False,
                    transform=transform)
test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

model = torch.load("sketch/models/mnist_cnn.pth")

model.eval()
test_loss = 0
correct = 0

for data, target in test_loader:
    data, target = data.to(device), target.to(device)

    img = data[0, :, :, :].cpu().numpy()
    plt.figure()
    plt.subplot(2, 3, 1)
    img_show = data[0, 0, :, :].cpu().numpy()
    img_show = img_show.squeeze()
    im = plt.imshow(img_show)
    plt.title('Original image')
    plt.colorbar(im)

    for lay in range(5):
        Q = compute_Q_obj.sample_q_train_ip(img, lay)
        plt.subplot(2, 3, 6-lay)
        im = plt.imshow(Q, cmap='gray')
        plt.title('Layer: '+str(5-lay))
        plt.colorbar(im)

    plt.suptitle('IP estimator')

    plt.figure()
    plt.subplot(2, 3, 1)
    img_show = data[0, 0, :, :].cpu().numpy()
    img_show = img_show.squeeze()
    im = plt.imshow(dct_2d(img_show))
    plt.title('Original image')
    plt.colorbar(im)
    
    for lay in range(5):
        Q = compute_Q_obj.sample_q_train_diag(img, lay)
        plt.subplot(2, 3, 6-lay)
        im = plt.imshow(Q, cmap='gray')
        plt.title('Layer: '+str(5-lay))
        plt.colorbar(im)
    plt.suptitle('Diag. estimator')

    plt.figure()
    plt.subplot(2, 3, 1)
    img_show = data[0, 0, :, :].cpu().numpy()
    img_show = img_show.squeeze()
    im = plt.imshow(dct_2d(img_show))
    plt.title('Original image')
    plt.colorbar(im)
    
    H_layers = np.zeros((5, 28*28, 28*28))
    est_ood = np.zeros((5, 1))
    for lay in range(5):
        H = compute_Q_obj.compute_exact_hessian(img, lay)
        est_ood[lay] = compute_Q_obj.compute_ood(img, lay).item()
        H_layers[lay, :, :] = H
        H_diag = (np.diag(H)).reshape((28, 28))
        plt.subplot(2, 3, 6-lay)
        im = plt.imshow(H_diag, cmap='gray')
        plt.title('Layer: '+str(5-lay))
        plt.colorbar(im)
    plt.suptitle('Exact Hessian diagonal')

    plt.figure()
    plt.subplot(1, 5, 1)
    for lay in range(5):
        H = H_layers[lay, :, :]
        plt.subplot(1, 5, 5-lay)
        im = plt.matshow(H, fignum=0)
        OOD = (np.sum(np.square(H)) - np.sum(np.square(np.diag(H)))) / (np.sum(np.square(H)))
        est_OOD = est_ood[lay]
        plt.title('Layer: '+str(5-lay))
        plt.xticks([])
        plt.yticks([])
    plt.show()
