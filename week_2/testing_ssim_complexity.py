import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_pytorch import compute_Q_ssim, compute_Q_msssim, compute_Q_brisque, compute_Q_lpips
import random
import time

N = (4, 8, 16)
n_cwd = 2
true_N = (1024, 1024)

compute_Q_obj_ssim = compute_Q_ssim(true_N=true_N, n_cwd=n_cwd, N=N[0], sampling_depth=128)
compute_Q_obj_ssim_j1 = compute_Q_ssim(true_N=true_N, n_cwd=n_cwd, N=N[0], sampling_depth=16)

path = 'Images/CLIC/Training/'
dirs = os.listdir(path)
num_images = 10
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

ind_image = -1

times = np.zeros((num_images, 1))
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()

    # measure the time
    start_time = time.time()
    Q_ssim_j1 = compute_Q_obj_ssim_j1.sample_q_diag_radamacher_train(img)
    end_time = time.time()
    times[ind_image] = end_time - start_time
    
# create boxplot of times
plt.figure()
plt.boxplot(times)
plt.ylabel('Time (s)', fontsize=16)
# set as xtick the ssim
plt.xticks([1], ['SSIM'], fontsize=16)
plt.yticks(fontsize=16)
plt.show()

    
    
