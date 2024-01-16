import torchvision
import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import datasets, transforms
import scipy
from compute_q.compute_Q_machines import compute_Q_machines


true_N = (224, 224)

compute_Q = compute_Q_machines(true_N, n_cwd=4, N=8)

# load some images
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# set the random seed
torch.manual_seed(3)

data_dir = 'Images/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

num_images = 5
plt.figure()
for i in range(num_images):
    al = iter(dataloaders['train'])
    img, _ = next(al)
    img = img[0, :, :, :].numpy()
    Q = compute_Q.sample_q_train(img)
    img = img.transpose((1, 2, 0))
    plt.subplot(num_images, 3, 1 + i*3)
    plt.imshow(img)
    plt.colorbar()
    plt.subplot(num_images, 3, 2 + i*3)
    plt.imshow(Q)
    plt.colorbar()  
    plt.subplot(num_images, 3, 3 + i*3)
    img = np.dot(img, [0.2989, 0.5870, 0.1140])
    plt.imshow(img*Q)
    plt.colorbar()
plt.show()

breakpoint()