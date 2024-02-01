import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim, compute_Q_msssim, compute_Q_brisque, compute_Q_lpips
from handling_q.q_ops import q_ops_ssim, q_ops_msssim, q_ops_brisque, q_ops_lpips
import random

N = (4, 8, 16)
n_cwd = 2
true_N = (1024, 1024)

compute_Q_obj_ssim = compute_Q_ssim(true_N=true_N, sampling_depth=32)
compute_Q_obj_ssim_j1 = compute_Q_ssim(true_N=true_N, sampling_depth=16)

q_ops_obj_ssim = q_ops_ssim(true_N=true_N, n_cwd=n_cwd, N=N)

path = 'Images/CLIC/Training/'
dirs = os.listdir(path)
num_images = 50
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

ind_image = -1

for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()

    Q_ssim = compute_Q_obj_ssim.sample_q_diag_radamacher(img)
    Q_ssim = q_ops_obj_ssim.normalize_q(Q_ssim)
    Q_ssim_j1 = compute_Q_obj_ssim_j1.sample_q_ip_radamacher(img)
    Q_ssim_j1 = q_ops_obj_ssim.normalize_q(Q_ssim_j1)

    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.imshow(Q_ssim, cmap='gray')
    plt.title('SSIM', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.imshow(Q_ssim_j1, cmap='gray')
    plt.title('SSIM, 1 shot', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    plt.suptitle('Diagonal estimator. Radamacher.', fontsize=18)
    plt.show()
    
    
