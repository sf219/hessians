import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_msssim, compute_Q_ssim, compute_Q_brisque, compute_Q_lpips
import random
import time
from jax.scipy.signal import convolve2d
import jax
import jax.numpy as jnp
import jax.scipy as jsp

#jax.default_device(jax.devices('cpu')[0])

true_N = (1024, 1024)

compute_Q_obj_ssim = compute_Q_ssim(true_N=true_N, sampling_depth=8)
compute_Q_obj_msssim = compute_Q_msssim(true_N=true_N, sampling_depth=8)
compute_Q_obj_brisque = compute_Q_brisque(true_N=true_N, sampling_depth=8)
compute_Q_oobj_lpips = compute_Q_lpips(true_N=true_N, sampling_depth=64)

path = 'Images/CLIC/Training/'
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
times_conv = np.zeros((num_images, 1))

for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()

    # measure the time
    start_time = time.time()
    Q_ssim_j1 = compute_Q_obj_ssim.sample_q_diag_radamacher(img)
    end_time = time.time()
    times_ssim[ind_image] = end_time - start_time
    
ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    start_time = time.time()
    Q_msssim_j1 = compute_Q_obj_msssim.sample_q_diag_radamacher(img)
    end_time = time.time()
    times_msssim[ind_image] = end_time - start_time

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    start_time = time.time()
    Q_msssim_j1 = compute_Q_obj_brisque.sample_q_diag_radamacher(img)
    end_time = time.time()
    times_brisque[ind_image] = end_time - start_time

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    start_time = time.time()
    Q_msssim_j1 = compute_Q_oobj_lpips.sample_q_diag_radamacher(img)
    end_time = time.time()
    times_lpips[ind_image] = end_time - start_time

truncate = 3.5
sigma = 1.5
#r = int(truncate * sigma + 0.5)  # radius as in ndimage
win_size = 16 #2*(2 * r + 1)
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
    img = jnp.array(img)
    conv_win = jnp.array(window)
    Q_ssim_j1_conv = jsp.signal.convolve(img, conv_win, mode='same')
    Q_ssim_j1_conv = jsp.signal.convolve(Q_ssim_j1_conv, conv_win, mode='same')
    Q_ssim_j1_conv = jsp.signal.convolve(Q_ssim_j1_conv, conv_win, mode='same')
    Q_ssim_j1_conv = jsp.signal.convolve(Q_ssim_j1_conv, conv_win, mode='same')
    Q_ssim_j1_conv = jsp.signal.convolve(Q_ssim_j1_conv, conv_win, mode='same')
    end_time = time.time()
    times_conv[ind_image] = end_time - start_time

times_ssim = times_ssim[1:]
times_conv = times_conv[1:]
times_msssim = times_msssim[1:]
times_brisque = times_brisque[1:]
times_lpips = times_lpips[1:]

times = np.concatenate((times_ssim, times_msssim, times_brisque, times_lpips, times_conv), axis=1)
plt.figure()
plt.boxplot(times)
plt.ylabel('Time (s)', fontsize=16)
# set as xtick the ssim
plt.xticks([1, 2, 3, 4, 5], ['SSIM', 'MS-SSIM', 'BRISQUE', 'LPIPS', '5 convolutions'], fontsize=16)
plt.yticks(fontsize=16)
plt.show()

    
    
