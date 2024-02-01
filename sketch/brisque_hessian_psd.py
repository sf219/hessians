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
num_images = 10
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

ind_image = -1

num_zeros_ssim = np.zeros((num_images, 256))
num_zeros_msssim = np.zeros((num_images, 256))
num_zeros_brisque = np.zeros((num_images, 256)) 
num_zeros_lpips = np.zeros((num_images, 256))   

for img_name in dirs:
    ind_image += 1
    if ind_image == 5:
        break
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    img = img/255*2
    
    compute_Q_obj_ssim.check_evaluation(img)
    compute_Q_obj_msssim.check_evaluation(img)
    compute_Q_obj_brisque.check_evaluation(img)
    compute_Q_obj_lpips.check_evaluation(img-1)

    compute_Q_obj_ssim.check_first_derivative(img)
    compute_Q_obj_msssim.check_first_derivative(img)
    compute_Q_obj_lpips.check_first_derivative(img)
    compute_Q_obj_brisque.check_first_derivative(img-1)

    Q_ssim = compute_Q_obj_ssim.compute_exact_hessian_cols(img)
    Q_msssim = compute_Q_obj_msssim.compute_exact_hessian_cols(img)       
    Q_brisque = compute_Q_obj_brisque.compute_exact_hessian_cols(img)
    Q_lpips = compute_Q_obj_lpips.compute_exact_hessian_cols(img-1)

    breakpoint()
    num_zeros_ssim[ind_image, 0] = np.sum(np.diag(Q_ssim) == 0)
    num_zeros_msssim[ind_image, 0] = np.sum(np.diag(Q_msssim) == 0)
    num_zeros_brisque[ind_image, 0] = np.sum(np.diag(Q_brisque) == 0)  
    num_zeros_lpips[ind_image, 0] = np.sum(np.diag(Q_lpips) == 0)

print('Prob. of zeros in SSIM: ', np.mean(num_zeros_ssim))
print('Prob. of zeros in MSSSIM: ', np.mean(num_zeros_msssim))
print('Prob. of zeros in BRISQUE: ', np.mean(num_zeros_brisque))
print('Prob. of zeros in LPIPS: ', np.mean(num_zeros_lpips))