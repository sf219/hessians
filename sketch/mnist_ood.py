from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from compute_q.compute_Q_machines import compute_Q_layer_mnist as compute_Q
import matplotlib.pyplot as plt
import numpy as np

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

num_samples = 25

est_ood = np.zeros((5, num_samples))
true_ood = np.zeros((5, num_samples))

iters = -1
for data, target in test_loader:
    iters += 1
    print('Iteration: ', iters, ' of ', num_samples)
    if iters == num_samples:
        break   

    data, target = data.to(device), target.to(device)

    img = data[0, :, :, :].cpu().numpy()
    
    for lay in range(5):
        H = compute_Q_obj.compute_exact_hessian(img, lay)
        true_ood[4-lay, iters] = (np.sum(np.square(H)) - np.sum(np.square(np.diag(H)))) / (np.sum(np.square(H)))
        est_ood[4-lay, iters] = compute_Q_obj.compute_ood(img, lay).item()

# box plot both

plt.figure()
plt.boxplot(est_ood.T)
plt.title('Estimated OOD')
# remember that the layers are inverted
plt.xticks(np.arange(1, 6), ('Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'), fontsize=16)

plt.figure()
plt.boxplot(true_ood.T)
plt.title('True OOD')
# remember that the layers are inverted
plt.xticks(np.arange(1, 6), ('Layer 1', 'Layer 2', 'Layer 3', 'Layer 4', 'Layer 5'), fontsize=16)

plt.show()


