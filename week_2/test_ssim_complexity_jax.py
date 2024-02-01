import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim
import random
import time
from scipy.signal import convolve2d

true_N = (512, 512)

compute_Q_obj_ssim = compute_Q_ssim(true_N=true_N, sampling_depth=8)

path = 'Images/CLIC/Training/'
dirs = os.listdir(path)
num_images = 10
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

ind_image = -1

times = np.zeros((num_images, 1))
times_conv = np.zeros((num_images, 1))

conv_win = np.ones((8, 8))

for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()

    # measure the time
    start_time = time.time()
    Q_ssim_j1 = compute_Q_obj_ssim.sample_q_diag_radamacher(img)
    end_time = time.time()
    times[ind_image] = end_time - start_time
    
    start_time = time.time()
    Q_ssim_j1_conv = convolve2d(img, conv_win, mode='same')
    Q_ssim_j1_conv = convolve2d(Q_ssim_j1_conv, conv_win, mode='same')
    Q_ssim_j1_conv = convolve2d(Q_ssim_j1_conv, conv_win, mode='same')
    Q_ssim_j1_conv = convolve2d(Q_ssim_j1_conv, conv_win, mode='same')
    end_time = time.time()
    times_conv[ind_image] = end_time - start_time

times = times[1:]
times_conv = times_conv[1:]

times = np.concatenate((times, times_conv), axis=1)
plt.figure()
plt.boxplot(times)
plt.ylabel('Time (s)', fontsize=16)
# set as xtick the ssim
plt.xticks([1, 2], ['SSIM', '4 convolutions'], fontsize=16)
plt.yticks(fontsize=16)
plt.show()

    
    
