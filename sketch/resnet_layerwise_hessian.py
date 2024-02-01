import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_pytorch import compute_Q_layer as compute_Q
from sklearn.cluster import KMeans
import random
from torchvision import transforms
from torchvision import datasets
import torch

N = (4, 8, 16)
n_cwd = 2
true_N = (224, 224)

compute_Q_obj = compute_Q(true_N=true_N, n_cwd=n_cwd, N=N[0])

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
    ]),
}

data_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

torch.random.manual_seed(3)

data_dir = 'Images/hymenoptera_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

num_images = 30
random.seed(0)

remove_first, _ = next(iter(dataloaders['val']))

for ind_image in range(num_images):
    print('Image: ', (ind_image), 'Number of images: ', num_images)
    img_org, _ = next(iter(dataloaders['val']))
    img_show = img_org[0, :, :, :].numpy()
    plt.figure()
    plt.subplot(2, 3, 1)
    img_show = img_show.transpose((1, 2, 0))
    plt.imshow(img_show)
    plt.title('Original image')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    titles = ['Block 1', 'Block 2', 'Block 3', 'Block 4', 'Avg. Pool']
    for lay in range(5):
        Q = compute_Q_obj.sample_q_diag_radamacher_train(img_org, lay)
        plt.subplot(2, 3, 6-lay)
        plt.imshow(Q)
        plt.title(titles[4-lay])
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
    plt.suptitle('Diagonal. Image: ' + str(ind_image))

    plt.figure()
    plt.subplot(2, 3, 1)
    plt.imshow(img_show)
    plt.title('Original image')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    titles = ['Block 1', 'Block 2', 'Block 3', 'Block 4', 'Avg. Pool']
    for lay in range(5):
        Q = compute_Q_obj.sample_q_ip_radamacher_train(img_org, lay)
        plt.subplot(2, 3, 6-lay)
        plt.imshow(Q)
        plt.title(titles[4-lay])
        plt.xticks([])
        plt.yticks([])
        plt.colorbar()
    plt.suptitle('Inner product. Image: ' + str(ind_image))
    plt.show()
