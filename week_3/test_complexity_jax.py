import os
#os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_msssim, compute_Q_ssim, compute_Q_brisque, compute_Q_lpips, compute_Q_niqe
import random
import time
from jax.scipy.signal import convolve2d
import jax
import jax.numpy as jnp
import jax.scipy as jsp
from utils.q_utils_jax import brisque_func, compute_LPIPS_gs, niqe_func
from iqa_funs.jax_ssim import jax_ssim, jax_msssim

true_N = (512, 768)

sampling_depth = 8
compute_Q_obj_ssim = compute_Q_ssim(true_N=true_N, sampling_depth=sampling_depth)
compute_Q_obj_msssim = compute_Q_msssim(true_N=true_N, sampling_depth=sampling_depth)
compute_Q_obj_brisque = compute_Q_brisque(true_N=true_N, sampling_depth=sampling_depth)
compute_Q_obj_lpips = compute_Q_lpips(true_N=true_N, sampling_depth=sampling_depth)
compute_Q_obj_niqe = compute_Q_niqe(true_N=true_N, sampling_depth=sampling_depth)

path = 'Images/KODAK/All/'
dirs = os.listdir(path)
num_images = 10
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

ind_image = -1

times_ssim = np.zeros((num_images, 1))
times_msssim = np.zeros((num_images, 1))
times_brisque = np.zeros((num_images, 1))
times_lpips = np.zeros((num_images, 1))
times_niqe = np.zeros((num_images, 1))
                      
times_hessian_ssim = np.zeros((num_images, 1))
times_hessian_msssim = np.zeros((num_images, 1))
times_hessian_brisque = np.zeros((num_images, 1))
times_hessian_lpips = np.zeros((num_images, 1))
times_hessian_niqe = np.zeros((num_images, 1))

for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()

    # measure the time
    start_time = time.time()
    Q_ssim_j1 = compute_Q_obj_ssim.sample_q_ip_radamacher(img)
    end_time = time.time()
    times_hessian_ssim[ind_image] = end_time - start_time
    
ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    start_time = time.time()
    Q_msssim_j1 = compute_Q_obj_msssim.sample_q_ip_radamacher(img)
    end_time = time.time()
    times_hessian_msssim[ind_image] = end_time - start_time

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    start_time = time.time()
    Q_msssim_j1 = compute_Q_obj_brisque.sample_q_ip_radamacher(img)
    end_time = time.time()
    times_hessian_brisque[ind_image] = end_time - start_time

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    start_time = time.time()
    Q_msssim_j1 = compute_Q_obj_lpips.sample_q_ip_radamacher(img)
    end_time = time.time()
    times_hessian_lpips[ind_image] = end_time - start_time

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    start_time = time.time()
    Q_msssim_j1 = compute_Q_obj_niqe.sample_q_ip_radamacher(img)
    end_time = time.time()
    times_hessian_niqe[ind_image] = end_time - start_time


truncate = 3.5
sigma = 1.5
r = int(truncate * sigma + 0.5)  # radius as in ndimage
win_size = 32
print('win_size: ', win_size)
x = jnp.linspace(-truncate, truncate, win_size)
window = jsp.stats.norm.pdf(x, scale=sigma) * jsp.stats.norm.pdf(x[:, None], scale=sigma)
window = window/jnp.sum(window)

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    start_time = time.time()
    img_1 = jnp.array(img).reshape((1, 1, img.shape[0], img.shape[1]))
    fwd_pass = jax_ssim(img_1, img_1)
    end_time = time.time()
    times_ssim[ind_image] = end_time - start_time
    start_time = time.time()
    img_1 = jnp.array(img).reshape((1, 1, img.shape[0], img.shape[1]))
    fwd_pass = jax_msssim(img_1, img_1)
    end_time = time.time()
    times_msssim[ind_image] = end_time - start_time
    start_time = time.time()
    img_1 = jnp.array(img).reshape((1, 1, img.shape[0], img.shape[1]))
    fwd_pass = brisque_func(img_1, img_1)
    end_time = time.time()
    times_brisque[ind_image] = end_time - start_time
    start_time = time.time()
    img_1 = jnp.array(img).reshape((1, 1, img.shape[0], img.shape[1]))
    fwd_pass = compute_LPIPS_gs(img_1, img_1)
    end_time = time.time()
    times_lpips[ind_image] = end_time - start_time
    start_time = time.time()
    img_1 = jnp.array(img).reshape((1, 1, img.shape[0], img.shape[1]))
    fwd_pass = niqe_func(img_1, img_1)
    end_time = time.time()
    times_niqe[ind_image] = end_time - start_time


times_ssim = times_ssim[1:]
times_msssim = times_msssim[1:]
times_brisque = times_brisque[1:]
times_lpips = times_lpips[1:]
times_niqe = times_niqe[1:]

times_hessian_ssim = times_hessian_ssim[1:]
times_hessian_msssim = times_hessian_msssim[1:]
times_hessian_brisque = times_hessian_brisque[1:]
times_hessian_lpips = times_hessian_lpips[1:]
times_hessian_niqe = times_hessian_niqe[1:]

# compute average and standard error for each
mean_ssim = np.mean(times_ssim)
std_ssim = np.std(times_ssim)
mean_msssim = np.mean(times_msssim)
std_msssim = np.std(times_msssim)
mean_brisque = np.mean(times_brisque)
std_brisque = np.std(times_brisque)
mean_lpips = np.mean(times_lpips)
std_lpips = np.std(times_lpips)
mean_niqe = np.mean(times_niqe)
std_niqe = np.std(times_niqe)

mean_hessian_ssim = np.mean(times_hessian_ssim)
std_hessian_ssim = np.std(times_hessian_ssim)
mean_hessian_msssim = np.mean(times_hessian_msssim)
std_hessian_msssim = np.std(times_hessian_msssim)
mean_hessian_brisque = np.mean(times_hessian_brisque)
std_hessian_brisque = np.std(times_hessian_brisque)
mean_hessian_lpips = np.mean(times_hessian_lpips)
std_hessian_lpips = np.std(times_hessian_lpips)
mean_hessian_niqe = np.mean(times_hessian_niqe)
std_hessian_niqe = np.std(times_hessian_niqe)

# print the results
print('SSIM: ', mean_ssim, '+-', std_ssim)
print('MS-SSIM: ', mean_msssim, '+-', std_msssim)
print('BRISQUE: ', mean_brisque, '+-', std_brisque)
print('LPIPS: ', mean_lpips, '+-', std_lpips)
print('NIQE: ', mean_niqe, '+-', std_niqe)

print('Hessian SSIM: ', mean_hessian_ssim, '+-', std_hessian_ssim)
print('Hessian MS-SSIM: ', mean_hessian_msssim, '+-', std_hessian_msssim)
print('Hessian BRISQUE: ', mean_hessian_brisque, '+-', std_hessian_brisque)
print('Hessian LPIPS: ', mean_hessian_lpips, '+-', std_hessian_lpips)
print('Hessian NIQE: ', mean_hessian_niqe, '+-', std_hessian_niqe)
    
    
