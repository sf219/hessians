import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_pytorch import compute_Q_ssim, compute_Q_msssim, compute_Q_brisque, compute_Q_lpips
from sklearn.cluster import KMeans
import random
from torchvision import transforms
from torchvision import datasets
import torch

N = (4, 8, 16)
n_cwd = 2
true_N = (128, 128)

compute_Q_obj_ssim = compute_Q_ssim(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_obj_msssim = compute_Q_msssim(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_obj_brisque = compute_Q_brisque(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_obj_lpips = compute_Q_lpips(true_N=true_N, n_cwd=n_cwd, N=N[0])

path = 'Images/CLIC/Training/'
dirs = os.listdir(path)
num_images = 50
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

ood_ssim = np.zeros((num_images, 1))
ood_msssim = np.zeros((num_images, 1))
ood_brisque = np.zeros((num_images, 1))
ood_lpips = np.zeros((num_images, 1))

ind_image = -1

for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    ood_ssim[ind_image] = compute_Q_obj_ssim.sample_ood(img)
    ood_msssim[ind_image] = compute_Q_obj_msssim.sample_ood(img)
    ood_brisque[ind_image] = compute_Q_obj_brisque.sample_ood(img)
    ood_lpips[ind_image] = compute_Q_obj_lpips.sample_ood(img)

    print('SSIM: ', ood_ssim[ind_image])
    print('MSSSIM: ', ood_msssim[ind_image])
    print('BRISQUE: ', ood_brisque[ind_image])
    print('LPIPS: ', ood_lpips[ind_image])

np.savez('ood_values.npz', ood_ssim=ood_ssim, ood_msssim=ood_msssim, ood_brisque=ood_brisque, ood_lpips=ood_lpips)
plt.figure()
plt.boxplot([ood_ssim, ood_msssim, ood_brisque, ood_lpips])
plt.xticks([1, 2, 3, 4], ['SSIM', 'MSSSIM', 'BRISQUE', 'LPIPS'], fontsize=16)
plt.yticks(fontsize=16)
plt.ylabel('OOD', fontsize=16)
plt.show()

