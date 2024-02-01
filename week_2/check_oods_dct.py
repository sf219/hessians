import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_msssim
import random


true_N = (128, 128)

compute_Q_obj = compute_Q_msssim(true_N=true_N, sampling_depth=64)

path = 'Images/CLIC/Testing/'
dirs = os.listdir(path)
num_images = 10
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]


ood_vec = np.zeros((num_images, 1))

ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()

    # measure the time
    ood_vec[ind_image] = compute_Q_obj.sample_ood(img)

N = 8
ood_vec_trans = np.zeros((num_images, 1))
ind_image = -1
for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()
    ood_vec_trans[ind_image] = compute_Q_obj.sample_ood_dct(img, N)

values = np.array([ood_vec, ood_vec_trans]).squeeze()
plt.figure()
plt.boxplot(values.T, labels=['MS-SSIM', 'Trans. MS-SSIM'])
plt.ylabel('OOD', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()