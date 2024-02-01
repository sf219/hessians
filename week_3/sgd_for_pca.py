import numpy as np
from utils.utils_lpit import read_image_resize_rect, ycbcr2rgb
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim, compute_Q_msssim, compute_Q_brisque, compute_Q_lpips
from handling_q.q_ops import q_ops_ssim, q_ops_msssim, q_ops_brisque, q_ops_lpips
import random
from week_3.density import tridiag_to_density
import jax

N = (4, 8, 16)
n_cwd = 2
true_N = (256, 256)

compute_Q_obj_ssim = compute_Q_ssim(true_N=true_N, sampling_depth=4)
compute_Q_obj_msssim = compute_Q_msssim(true_N=true_N, sampling_depth=4)
compute_Q_obj_brisque = compute_Q_brisque(true_N=true_N, sampling_depth=4)
compute_Q_obj_lpips = compute_Q_lpips(true_N=true_N, sampling_depth=4)

q_ops_obj_ssim = q_ops_ssim(true_N=true_N, n_cwd=n_cwd, N=N[0], nqs=6)
q_ops_obj_msssim = q_ops_msssim(true_N=true_N, n_cwd=n_cwd, N=N[0], nqs=6)
q_ops_obj_brisque = q_ops_brisque(true_N=true_N, n_cwd=n_cwd, N=N[0], nqs=6)
q_ops_obj_lpips = q_ops_lpips(true_N=true_N, n_cwd=n_cwd, N=N[0], nqs=6)

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
    img_col = ycbcr2rgb(img)
    img = img[:, :, 0].squeeze()
    
    rng = jax.random.PRNGKey(0)
    rng, split = jax.random.split(rng)    
    dim = true_N[0]*true_N[1]
    order = 200

    w_ssim = compute_Q_obj_ssim.get_sgd_hvp(img, 1000)
    w_msssim = compute_Q_obj_msssim.get_sgd_hvp(img, 1000)
    w_brisque = compute_Q_obj_brisque.get_sgd_hvp(img, 1000)
    w_lpips = compute_Q_obj_lpips.get_sgd_hvp(img, 1000)

        
    plt.figure()
    plt.subplot(1, 5, 1)
    plt.imshow(img_col)
    plt.axis('off')
    plt.subplot(1, 5, 2)
    plt.imshow(np.abs(w_ssim.reshape(true_N, order='F')), cmap='gray')
    plt.title('SSIM')
    plt.axis('off') 
    plt.subplot(1, 5, 3)
    plt.imshow(np.abs(w_msssim.reshape(true_N, order='F')), cmap='gray')
    plt.title('MSSSIM')
    plt.axis('off')
    plt.subplot(1, 5, 4)
    plt.imshow(np.abs(w_brisque.reshape(true_N, order='F')), cmap='gray')
    plt.title('BRISQUE')
    plt.axis('off')
    plt.subplot(1, 5, 5)
    plt.imshow(np.abs(w_lpips.reshape(true_N, order='F')), cmap='gray')
    plt.title('LPIPS')
    plt.axis('off')

    plt.figure()
    plt.subplot(1, 5, 1)
    plt.imshow(img_col)
    plt.axis('off')
    plt.subplot(1, 5, 2)
    plt.imshow((w_ssim.reshape(true_N, order='F')), cmap='gray')
    plt.title('SSIM')
    plt.axis('off') 
    plt.subplot(1, 5, 3)
    plt.imshow((w_msssim.reshape(true_N, order='F')), cmap='gray')
    plt.title('MSSSIM')
    plt.axis('off')
    plt.subplot(1, 5, 4)
    plt.imshow((w_brisque.reshape(true_N, order='F')), cmap='gray')
    plt.title('BRISQUE')
    plt.axis('off')
    plt.subplot(1, 5, 5)
    plt.imshow((w_lpips.reshape(true_N, order='F')), cmap='gray')
    plt.title('LPIPS')
    plt.axis('off')

    plt.show()
    breakpoint()
    
    
