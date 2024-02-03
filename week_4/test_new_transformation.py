import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_msssim, compute_Q_ssim, compute_Q_brisque, compute_Q_lpips
import random

true_N = (256, 256)

compute_Q_obj_ssim = compute_Q_ssim(true_N=true_N, sampling_depth=32)
compute_Q_obj_msssim = compute_Q_msssim(true_N=true_N, sampling_depth=32)
compute_Q_obj_brisque = compute_Q_brisque(true_N=true_N, sampling_depth=32)
compute_Q_obj_lpips = compute_Q_lpips(true_N=true_N, sampling_depth=32)

path = 'Images/CLIC/Testing/'
dirs = os.listdir(path)
num_images = 50
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

N = 8

ood_vec_ssim = np.zeros((num_images, 1))
ood_vec_msssim = np.zeros((num_images, 1))
ood_vec_brisque = np.zeros((num_images, 1))
ood_vec_lpips = np.zeros((num_images, 1))

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    ood_vec_ssim[ind_image] = compute_Q_obj_ssim.sample_ood(img)

tmp = np.load('week_2/data/eigenvectors_8_msssim_512.npz')
eig_vecs = tmp['eig_vecs']
eig_vals = tmp['eig_vals']
ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    ood_vec_msssim[ind_image] = compute_Q_obj_msssim.sample_ood_transformation(img, eig_vecs)

tmp = np.load('week_2/data/eigenvectors_8_brisque_512.npz')
eig_vecs = tmp['eig_vecs']
eig_vals = tmp['eig_vals']


ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    ood_vec_brisque[ind_image] = compute_Q_obj_brisque.sample_ood_transformation(img, eig_vecs)

tmp = np.load('week_2/data/eigenvectors_8_lpips_512.npz')
eig_vecs = tmp['eig_vecs']
eig_vals = tmp['eig_vals']

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    ood_vec_lpips[ind_image] = compute_Q_obj_lpips.sample_ood_transformation(img, eig_vecs)

values = np.array([ood_vec_ssim, ood_vec_msssim, ood_vec_brisque, ood_vec_lpips]).squeeze()
plt.figure()
plt.boxplot(values.T, labels=['SSIM', 'MS-SSIM', 'BRISQUE', 'LPIPS'])
plt.ylabel('OOD', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()