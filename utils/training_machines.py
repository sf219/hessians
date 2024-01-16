import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_machines import compute_Q_machines as compute_Q
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
        transforms
        .Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

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

q_vecs_4 = np.zeros((num_images, true_N[0]//N[0], true_N[1]//N[0], N[0]*N[0]))
q_vecs_8 = np.zeros((num_images, true_N[0]//N[1], true_N[1]//N[1], N[1]*N[1]))
q_vecs_16 = np.zeros((num_images, true_N[0]//N[2], true_N[1]//N[2], N[2]*N[2]))

for ind_image in range(num_images):
    print('Image: ', (ind_image), 'Number of images: ', num_images)
    img, _ = next(iter(dataloaders['train']))
    img = img[0, :, :, :].numpy()
    Q = compute_Q_obj.sample_q_train(img)

    # iterate over all blocks of img
    for i in range(0, true_N[0]//N[0]):
        for j in range(0, true_N[1]//N[0]):
            ravel_q = Q[i:i+N[0], j:j+N[0]].ravel('F')
            q_vecs_4[ind_image, i, j, :] = np.array(ravel_q)

    for i in range(0, true_N[0]//N[1]):
        for j in range(0, true_N[1]//N[1]):
            ravel_q = Q[i:i+N[1], j:j+N[1]].ravel('F')
            q_vecs_8[ind_image, i, j, :] = np.array(ravel_q)

    for i in range(0, true_N[0]//N[2]):
        for j in range(0, true_N[1]//N[2]):
            ravel_q = Q[i:i+N[2], j:j+N[2]].ravel('F')
            q_vecs_16[ind_image, i, j, :] = np.array(ravel_q)

    """
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(Q, cmap='gray')
    plt.colorbar()
    plt.show()
    """
n_cwds = np.arange(2, 11)

for n_cwd in n_cwds:
    q_vecs_4_tmp = q_vecs_4.transpose(3, 0, 1, 2)
    q_batch = q_vecs_4_tmp.reshape(N[0]**2, -1)
    # kmeans clustering
    kmeans = KMeans(n_clusters=n_cwd, random_state=0).fit(q_batch.T)
    # compute the centroids
    centroids = kmeans.cluster_centers_
    centroids = centroids.reshape(n_cwd, N[0], N[0])

    target = compute_Q_obj.name_target()

    str_save = 'data/centroids/centroids_' + target + '_' + str(n_cwd) + '_' + str(true_N) + '_' + str(N[0]) + '.npy'

    np.save(str_save, centroids)

    q_vecs_8_tmp = q_vecs_8.transpose(3, 0, 1, 2)
    q_batch = q_vecs_8_tmp.reshape(N[1]**2, -1)
    # kmeans clustering
    kmeans = KMeans(n_clusters=n_cwd, random_state=0).fit(q_batch.T)
    # compute the centroids
    centroids = kmeans.cluster_centers_
    centroids = centroids.reshape(n_cwd, N[1], N[1])

    str_save = 'data/centroids/centroids_' + target + '_' + str(n_cwd) + '_' + str(true_N) + '_' + str(N[1]) + '.npy'

    np.save(str_save, centroids)

    q_vecs_16_tmp = q_vecs_16.transpose(3, 0, 1, 2)
    q_batch = q_vecs_16_tmp.reshape(N[2]**2, -1)
    # kmeans clustering
    kmeans = KMeans(n_clusters=n_cwd, random_state=0).fit(q_batch.T)
    # compute the centroids
    centroids = kmeans.cluster_centers_
    centroids = centroids.reshape(n_cwd, N[2], N[2])

    mean_val = np.mean(centroids, axis=(1, 2))
    order = np.argsort(mean_val)
    centroids = centroids[order, :, :]

    str_save = 'data/centroids/centroids_' + target + '_' + str(n_cwd) + '_' + str(true_N) + '_' + str(N[2]) + '.npy'
    np.save(str_save, centroids)


    # plot the centroids
    fig, axs = plt.subplots(1, n_cwd, figsize=(12, 4))
    for i in range(n_cwd):
        im = axs[i].imshow(centroids[i, :, :], cmap='gray')
        axs[i].set_title('Centroid {}'.format(i+1))
        axs[i].axis('off')
        axs[i].set_aspect('equal')
        # add colorbar
        fig.colorbar(im, ax=axs[i])
plt.show()

