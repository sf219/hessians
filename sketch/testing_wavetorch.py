import torch
import torch_dct as dct
import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_dct_pp  import compute_Q_ssim_dct, compute_Q_msssim_dct, compute_Q_lpips_dct, compute_Q_brisque_dct
from compute_q.compute_Q_dwt_pp  import compute_Q_ssim_dwt, compute_Q_msssim_dwt, compute_Q_lpips_dwt, compute_Q_brisque_dwt
from compute_q.compute_Q_pytorch import compute_Q_ssim, compute_Q_msssim, compute_Q_lpips, compute_Q_brisque
import random
import torch

N = (4, 8, 16)
n_cwd = 2
true_N = (128, 128)

compute_Q_obj_ssim_dwt = compute_Q_ssim_dwt(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_obj_msssim_dwt = compute_Q_msssim_dwt(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_obj_brisque_dwt = compute_Q_brisque_dwt(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_obj_lpips_dwt = compute_Q_lpips_dwt(true_N=true_N, n_cwd=n_cwd, N=N[0])

compute_Q_obj_ssim = compute_Q_ssim(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_obj_msssim = compute_Q_msssim(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_obj_brisque = compute_Q_brisque(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_obj_lpips = compute_Q_lpips(true_N=true_N, n_cwd=n_cwd, N=N[0])


path = 'Images/CLIC/Training/'
dirs = os.listdir(path)
num_images = 20
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

ood_ssim = np.zeros((num_images, 1))
ood_msssim = np.zeros((num_images, 1))
ood_brisque = np.zeros((num_images, 1))
ood_lpips = np.zeros((num_images, 1))

ood_ssim_dwt = np.zeros((num_images, 1))
ood_msssim_dwt = np.zeros((num_images, 1))
ood_brisque_dwt = np.zeros((num_images, 1))
ood_lpips_dwt = np.zeros((num_images, 1))

ind_image = -1

for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    ood_ssim[ind_image] = compute_Q_obj_ssim.sample_half_ood(img)
    ood_msssim[ind_image] = compute_Q_obj_msssim.sample_half_ood(img)
    ood_brisque[ind_image] = compute_Q_obj_brisque.sample_half_ood(img)
    ood_lpips[ind_image] = compute_Q_obj_lpips.sample_half_ood(img)

    ood_ssim_dwt[ind_image] = compute_Q_obj_ssim_dwt.sample_half_ood(img)
    ood_msssim_dwt[ind_image] = compute_Q_obj_msssim_dwt.sample_half_ood(img)
    ood_brisque_dwt[ind_image] = compute_Q_obj_brisque_dwt.sample_half_ood(img)
    ood_lpips_dwt[ind_image] = compute_Q_obj_lpips_dwt.sample_half_ood(img)


    print('SSIM: ', ood_ssim[ind_image])
    print('MSSSIM: ', ood_msssim[ind_image])
    print('BRISQUE: ', ood_brisque[ind_image])
    print('LPIPS: ', ood_lpips[ind_image])

    print('SSIM DWT: ', ood_ssim_dwt[ind_image])
    print('MSSSIM DWT: ', ood_msssim_dwt[ind_image])
    print('BRISQUE DWT: ', ood_brisque_dwt[ind_image])
    print('LPIPS DWT: ', ood_lpips_dwt[ind_image])
# plot box plots

oods = np.asarray([ood_ssim, ood_msssim, ood_brisque, ood_lpips])
oods_dwt = np.asarray([ood_ssim_dwt, ood_msssim_dwt, ood_brisque_dwt, ood_lpips_dwt])

plt.figure()
plt.boxplot(oods.T)
plt.xticks([1, 2, 3, 4], ['SSIM', 'MSSSIM', 'BRISQUE', 'LPIPS'])
plt.ylabel('OOD Score')
plt.title('OOD Scores for CLIC Images')

plt.figure()
plt.boxplot(oods_dwt.T)
plt.xticks([1, 2, 3, 4], ['SSIM', 'MSSSIM', 'BRISQUE', 'LPIPS'])
plt.ylabel('OOD Score DWT')
plt.title('OOD Scores for CLIC Images')

plt.show()