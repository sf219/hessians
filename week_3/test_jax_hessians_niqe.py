import numpy as np
from utils.utils_lpit import read_image_resize_rect, ycbcr2rgb
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim, compute_Q_msssim, compute_Q_brisque, compute_Q_lpips, compute_Q_niqe
from handling_q.q_ops import q_ops_ssim, q_ops_msssim, q_ops_brisque, q_ops_lpips, q_ops_niqe
import random

N = (4, 8, 16)
n_cwd = 2
true_N = (288, 288)

compute_Q_obj_ssim = compute_Q_ssim(true_N=true_N, sampling_depth=4)
compute_Q_obj_msssim = compute_Q_msssim(true_N=true_N, sampling_depth=32)
compute_Q_obj_brisque = compute_Q_brisque(true_N=true_N, sampling_depth=32)
compute_Q_obj_lpips = compute_Q_lpips(true_N=true_N, sampling_depth=32)
compute_Q_obj_niqe = compute_Q_niqe(true_N=true_N, sampling_depth=32)

q_ops_obj_ssim = q_ops_ssim(true_N=true_N, n_cwd=n_cwd, N=N[0])
q_ops_obj_msssim = q_ops_msssim(true_N=true_N, n_cwd=n_cwd, N=N[0])
q_ops_obj_brisque = q_ops_brisque(true_N=true_N, n_cwd=n_cwd, N=N[0])
q_ops_obj_lpips = q_ops_lpips(true_N=true_N, n_cwd=n_cwd, N=N[0])
q_ops_obj_niqe = q_ops_niqe(true_N=true_N, n_cwd=n_cwd, N=N[0])

path = 'Images/CLIC/pretty/'
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
    img_col = ycbcr2rgb(img)
    img = img[:, :, 0].squeeze()

    Q_niqe = compute_Q_obj_niqe.sample_q_diag_radamacher(img)
    Q_ssim = compute_Q_obj_ssim.sample_q_diag_radamacher(img)
    Q_msssim = compute_Q_obj_msssim.sample_q_diag_radamacher(img)
    Q_brisque = compute_Q_obj_brisque.sample_q_diag_radamacher(img)
    Q_lpips = compute_Q_obj_lpips.sample_q_diag_radamacher(img)

    Q_ssim = q_ops_obj_ssim.normalize_q(Q_ssim)
    Q_msssim = q_ops_obj_msssim.normalize_q(Q_msssim)
    Q_brisque = q_ops_obj_brisque.normalize_q(Q_brisque)
    Q_lpips = q_ops_obj_lpips.normalize_q(Q_lpips)
    Q_niqe = q_ops_obj_niqe.normalize_q(Q_niqe)

    top=0.673
    bottom=0.061
    left=0.009
    right=0.983
    hspace=0.2
    wspace=0.0
    
    plt.figure(figsize=(16.5, 3))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)

    plt.subplot(1, 6, 1)
    im = plt.imshow(img_col)
    plt.title('Image', fontsize=16)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 6, 2)
    plt.imshow(Q_ssim, cmap='gray')
    plt.title('SSIM', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.subplot(1, 6, 3)
    plt.imshow(Q_msssim, cmap='gray')
    plt.title('MS-SSIM', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    plt.subplot(1, 6, 5)
    plt.imshow(Q_brisque, cmap='gray')
    plt.title('BRISQUE', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.subplot(1, 6, 4)
    plt.imshow(Q_lpips, cmap='gray')
    plt.title('LPIPS', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.subplot(1, 6, 6)
    plt.imshow(Q_niqe, cmap='gray')
    plt.title('NIQE', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.suptitle('Diagonal estimator', fontsize=18)

    plt.tight_layout()

    Q_ssim = compute_Q_obj_ssim.sample_q_ip_radamacher(img)
    Q_msssim = compute_Q_obj_msssim.sample_q_ip_radamacher(img)
    Q_brisque = compute_Q_obj_brisque.sample_q_ip_radamacher(img)
    Q_lpips = compute_Q_obj_lpips.sample_q_ip_radamacher(img)
    Q_niqe = compute_Q_obj_niqe.sample_q_ip_radamacher(img)

    Q_ssim = q_ops_obj_ssim.normalize_q(Q_ssim)
    Q_msssim = q_ops_obj_msssim.normalize_q(Q_msssim)
    Q_brisque = q_ops_obj_brisque.normalize_q(Q_brisque)
    Q_lpips = q_ops_obj_lpips.normalize_q(Q_lpips)
    Q_niqe = q_ops_obj_niqe.normalize_q(Q_niqe)

    plt.figure(figsize=(16.5, 3))
    plt.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace, wspace=wspace)
    
    plt.subplot(1, 6, 1)
    plt.imshow(img_col)
    plt.xticks([])
    plt.yticks([])
    plt.title('Image', fontsize=16)
    
    plt.subplot(1, 6, 2)
    plt.imshow(Q_ssim, cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title('SSIM', fontsize=16)
    
    plt.subplot(1, 6, 3)
    plt.imshow(Q_msssim, cmap='gray')
    plt.colorbar()  
    plt.xticks([])
    plt.yticks([])
    plt.title('MS-SSIM', fontsize=16)
    
    plt.subplot(1, 6, 5)
    plt.imshow(Q_brisque, cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title('BRISQUE', fontsize=16)

    plt.subplot(1, 6, 4)
    plt.imshow(Q_lpips, cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title('LPIPS', fontsize=16)

    plt.subplot(1, 6, 6)
    plt.imshow(Q_niqe, cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title('NIQE', fontsize=16)

    plt.suptitle('Column-norm estimator', fontsize=18)

    plt.tight_layout()

plt.show()
    
    
