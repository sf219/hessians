import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_lpips as compute_Q
from handling_q.q_ops import q_ops_lpips as q_ops
from sklearn.cluster import KMeans
import random

N = (4, 8, 16)
n_cwd = 8
true_N = (512, 512)
nqs = 6

compute_Q_obj = compute_Q(true_N=true_N, sampling_depth=16)
q_ops_obj = q_ops(true_N=true_N, n_cwd=n_cwd, N=N[0], nqs=nqs)

path = 'Images/CLIC/Training/'
dirs = os.listdir(path)
num_images = 100
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

q_vecs_4 = np.zeros((len(dirs), true_N[0]//N[0], true_N[1]//N[0], N[0]*N[0]))
q_vecs_8 = np.zeros((len(dirs), true_N[0]//N[1], true_N[1]//N[1], N[1]*N[1]))
q_vecs_16 = np.zeros((len(dirs), true_N[0]//N[2], true_N[1]//N[2], N[2]*N[2]))

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    Q = compute_Q_obj.sample_q(img)
    Q = q_ops_obj.normalize_q(Q)

    if np.isnan(Q).any():
        print('Nan in Q')
        continue
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

n_cwds = [2, 3, 4, 5, 6, 7, 8]

q_vecs_4_tmp = q_vecs_4
q_vecs_8_tmp = q_vecs_8
q_vecs_16_tmp = q_vecs_16

for n_cwd in n_cwds:

    q_vecs_4 = q_vecs_4_tmp.transpose(3, 0, 1, 2)
    # remove nans from q_vecs_4
    q_vecs_4 = q_vecs_4[~np.isnan(q_vecs_4).any(axis=(1, 2, 3))]
    q_batch = q_vecs_4.reshape(N[0]**2, -1)

    # kmeans clustering
    kmeans = KMeans(n_clusters=n_cwd, random_state=0).fit(q_batch.T)
    # compute the centroids
    centroids = kmeans.cluster_centers_
    centroids = centroids.reshape(n_cwd, N[0], N[0])

    target = q_ops_obj.name_target()

    str_save = 'data/centroids/centroids_' + target + '_' + str(n_cwd) + '_' + str(true_N) + '_' + str(N[0]) + '.npy'

    np.save(str_save, centroids)

    q_vecs_8 = q_vecs_8_tmp.transpose(3, 0, 1, 2)
    q_vecs_8 = q_vecs_8[~np.isnan(q_vecs_8).any(axis=(1, 2, 3))]
    q_batch = q_vecs_8.reshape(N[1]**2, -1)
    # kmeans clustering
    kmeans = KMeans(n_clusters=n_cwd, random_state=0).fit(q_batch.T)
    # compute the centroids
    centroids = kmeans.cluster_centers_
    centroids = centroids.reshape(n_cwd, N[1], N[1])

    str_save = 'data/centroids/centroids_' + target + '_' + str(n_cwd) + '_' + str(true_N) + '_' + str(N[1]) + '.npy'

    np.save(str_save, centroids)

    q_vecs_16 = q_vecs_16_tmp.transpose(3, 0, 1, 2)
    q_vecs_16 = q_vecs_16[~np.isnan(q_vecs_16).any(axis=(1, 2, 3))]
    q_batch = q_vecs_16.reshape(N[2]**2, -1)
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