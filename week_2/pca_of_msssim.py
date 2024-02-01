import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim, compute_Q_msssim, compute_Q_brisque, compute_Q_lpips
import random
import time
from scipy.fftpack import dct


true_N = (512, 512)

compute_Q_obj = compute_Q_lpips(true_N=true_N, sampling_depth=64)

path = 'Images/CLIC/Training/'
dirs = os.listdir(path)
num_images = 100
random.seed(1)
random.shuffle(dirs)
dirs = dirs[:num_images]

ind_image = -1

N = 8
times = np.zeros((num_images, 1))
cov = np.zeros((N**2, N**2))
mean = np.zeros((N**2, 1))
ood_vec = np.zeros((num_images, 1))
n_reps = 1

num_samples = 0
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()

    # measure the time

    for k in range(n_reps):
        Q_mtx = compute_Q_obj.just_sample_Q(img)
        #ood_vec[ind_image] = compute_Q_obj.sample_ood(img)
        for i in range(true_N[0]//N):
            for j in range(true_N[1]//N):
                block = Q_mtx[i*N:(i+1)*N, j*N:(j+1)*N].ravel('F')
                num_samples += 1
                if num_samples == 1:
                    var = np.mean(np.square(block))
                    mean = block
                    cov = np.outer(block, block)/var
                else:
                    mean += 1/(num_samples)*(block - mean)
                    var = np.mean(np.square(block-mean))
                    cov += 1/(num_samples-1)*(np.outer(block - mean, block - mean)/var - cov)

cov = (cov + cov.T)/2
eig_vals, eig_vecs = np.linalg.eigh(cov)
D = dct(np.eye(N**2), norm='ortho', axis=0).T

sort_ind = np.argsort(eig_vals)[::-1]
eig_vecs = eig_vecs[:, sort_ind]

signs = np.sign(np.diag(D.T @ eig_vecs))
eig_vecs = eig_vecs * signs

name = 'week_2/data/eigenvectors_' + str(N) + '_' + 'lpips_' + str(true_N[0]) + '.npz'

np.savez(name, eig_vecs=eig_vecs, eig_vals=eig_vals)

plt.figure()
plt.boxplot(ood_vec)
plt.ylabel('OOD', fontsize=16)
plt.xticks([1], ['SSIM'], fontsize=16)
plt.yticks(fontsize=16)

plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.plot(eig_vecs[:, i])
    plt.title('Eigenvector '+str(i), fontsize=16)
    
plt.figure()
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(eig_vecs[:, i].reshape((N, N)), cmap='gray')
    plt.title('Eigenvector '+str(i), fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

plt.show()