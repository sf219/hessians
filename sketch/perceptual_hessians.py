import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_pytorch import compute_Q_ssim, compute_Q_msssim, compute_Q_brisque, compute_Q_lpips
import random

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

ind_image = -1

for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()

    
    compute_Q_obj_ssim.check_evaluation(img)
    compute_Q_obj_msssim.check_evaluation(img)
    compute_Q_obj_brisque.check_evaluation(img)
    compute_Q_obj_lpips.check_evaluation(img)

    compute_Q_obj_ssim.check_first_derivative(img)
    compute_Q_obj_msssim.check_first_derivative(img)
    compute_Q_obj_lpips.check_first_derivative(img)
    compute_Q_obj_brisque.check_first_derivative(img)

    Q_ssim = compute_Q_obj_ssim.sample_q_diag_radamacher_train(img)
    Q_msssim = compute_Q_obj_msssim.sample_q_diag_radamacher_train(img)
    Q_brisque = compute_Q_obj_brisque.sample_q_diag_radamacher_train(img)
    Q_lpips = compute_Q_obj_lpips.sample_q_diag_radamacher_train(img)


    plt.figure()
    plt.subplot(1, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.subplot(1, 5, 2)
    plt.imshow(Q_ssim, cmap='gray')
    plt.title('SSIM', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.subplot(1, 5, 3)
    plt.imshow(Q_msssim, cmap='gray')
    plt.title('MS-SSIM', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    
    plt.subplot(1, 5, 4)
    plt.imshow(Q_brisque, cmap='gray')
    plt.title('BRISQUE', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()

    plt.subplot(1, 5, 5)
    plt.imshow(Q_lpips, cmap='gray')
    plt.title('LPIPS', fontsize=16)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    plt.suptitle('Diagonal estimator. Gaussian.')

    Q_ssim = compute_Q_obj_ssim.sample_q_ip_gaussian_train(img)
    Q_msssim = compute_Q_obj_msssim.sample_q_ip_gaussian_train(img)
    Q_brisque = compute_Q_obj_brisque.sample_q_ip_gaussian_train(img)
    Q_lpips = compute_Q_obj_lpips.sample_q_ip_gaussian_train(img)

    plt.figure()
    
    plt.subplot(1, 5, 1)
    plt.imshow(img, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar()
    plt.title('Original', fontsize=16)
    
    plt.subplot(1, 5, 2)
    plt.imshow(Q_ssim, cmap='gray')
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.title('SSIM', fontsize=16)
    
    plt.subplot(1, 5, 3)
    plt.imshow(Q_msssim, cmap='gray')
    plt.colorbar()  
    plt.xticks([])
    plt.yticks([])
    plt.title('MS-SSIM', fontsize=16)
    
    plt.subplot(1, 5, 4)
    plt.imshow(Q_brisque, cmap='gray')
    plt.title('BRISQUE', fontsize=16)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    plt.subplot(1, 5, 5)
    plt.imshow(Q_lpips, cmap='gray')
    plt.title('LPIPS', fontsize=16)
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])
    plt.suptitle('IP estimator. Gaussian.', fontsize=18)
    plt.show()
    
    
