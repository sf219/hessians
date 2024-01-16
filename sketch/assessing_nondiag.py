import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_pytorch import compute_Q_ssim
from compute_q.compute_Q_pytorch import compute_Q_msssim
from compute_q.compute_Q_pytorch import compute_Q_brisque
from compute_q.compute_Q_pytorch import compute_Q_lpips
from compute_q.compute_Q_pytorch import compute_Q_clip
from sklearn.cluster import KMeans
import random

N = (4, 8, 16) 
n_cwd = 4
true_N = (256, 256)

compute_Q_ssim_obj = compute_Q_ssim(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_msssim_obj = compute_Q_msssim(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_brisque_obj = compute_Q_brisque(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_lpips_obj = compute_Q_lpips(true_N=true_N, n_cwd=n_cwd, N=N[0])
compute_Q_clip_obj = compute_Q_clip(true_N=true_N, n_cwd=n_cwd, N=N[0])

path = 'Images/CLIC/Training/'
dirs = os.listdir(path)
num_images = 1
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

ind_image = -1

error_vec_ssim = np.zeros((len(dirs),))
error_vec_msssim = np.zeros((len(dirs),))
error_vec_brisque = np.zeros((len(dirs),))
error_vec_lpips = np.zeros((len(dirs),))
error_vec_clip = np.zeros((len(dirs),))

for img_name in dirs:
    ind_image += 1
    print('Image: ', (ind_image), 'Number of images: ', (len(dirs)))
    img, depth = read_image_resize_rect(path+img_name, true_N)
    img = img[:, :, 0].squeeze()

    Q_had_1 = compute_Q_ssim_obj.sample_q_train_had(img)
    Q_stoch_1 = compute_Q_ssim_obj.sample_q_train_stoch(img)
    error_vec_ssim[ind_image] = np.mean(np.abs(Q_stoch_1 - Q_had_1))

    print('SSIM done')

    Q_had_2 = compute_Q_msssim_obj.sample_q_train_had(img)
    Q_stoch_2 = compute_Q_msssim_obj.sample_q_train_stoch(img)
    error_vec_msssim[ind_image] = np.mean(np.abs(Q_stoch_2 - Q_had_2))

    print('MSSSIM done')

    Q_had_3 = compute_Q_brisque_obj.sample_q_train_had(img)
    Q_stoch_3 = compute_Q_brisque_obj.sample_q_train_stoch(img)
    error_vec_brisque[ind_image] = np.mean(np.abs(Q_stoch_3 - Q_had_3))

    print('BRISQUE done')

    Q_had_4 = compute_Q_lpips_obj.sample_q_train_had(img)
    Q_stoch_4 = compute_Q_lpips_obj.sample_q_train_stoch(img)
    error_vec_lpips[ind_image] = np.mean(np.abs(Q_stoch_4 - Q_had_4))

    print('LPIPS done')


plt.figure()
# do a boxplot with all the error_vecs
plt.boxplot([error_vec_ssim, error_vec_msssim, error_vec_brisque, error_vec_lpips])
plt.xticks([1, 2, 3, 4], ['SSIM', 'MSSSIM', 'BRISQUE', 'LPIPS'], fontsize=20)
# set yticks fontsize to 20 as well
plt.yticks(fontsize=20)
plt.ylabel('Error', fontsize=20)
plt.title('Error of stochastic vs. Hadamard Hessian', fontsize=20)
plt.show()