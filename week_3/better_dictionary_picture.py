from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG
from utils.utils_lpit import read_image_resize_rect, ycbcr2rgb
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim as compute_Q_class
from handling_q.q_ops import q_ops_ssim as compute_ops_class

import random
import torch
from utils.q_utils import ssim_func as ssim_mod
from utils.q_utils import ms_ssim_func as ms_ssim_mod
import numpy as np

#ssim_mod = SSIM(data_range=2.0, size_average=True, channel=1)
#ms_ssim_mod = MS_SSIM(data_range=2.0, size_average=True, channel=1)

#ssim_func = lambda x, y: -10*np.log10(ssim_mod(x, y).item())
#ms_ssim_func = lambda x, y: -10*np.log10(ms_ssim_mod(x, y).item())

ssim_func = lambda x, y: ssim_mod(x, y)
ms_ssim_func = lambda x, y: ms_ssim_mod(x, y)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_cwd = 4
true_N = (512, 512)
nqs = 6
N = 8

###### REMEMBER THAT NSQJPEG IS MODIFIED TO NOT INCLUDE RLGR ENCODING ######
###### 
flag_uniform = True

compute_Q_obj = compute_Q_class(true_N, sampling_depth=32)
q_ops_obj = compute_ops_class(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
nsqjpeg = NSQJPEG(compute_Q_obj, q_ops_obj, nqs=nqs, N=N, uniform=flag_uniform)
jpeg = JPEG(nqs, uniform=flag_uniform, N=8)

path = 'Images/CLIC/Training/'
dirs = os.listdir(path)
num_images = 20
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

for i in range(num_images):
    fullpath = os.path.join(path,dirs[i])  
    img, depth = read_image_resize_rect(fullpath, true_N)
    #img = img[:, :, 0]
    #depth = 1

    jpeg.set_Q(img)
    nsqjpeg.set_Q(img, one_depth=True)
    nsqjpeg.set_basis()

    img_show = ycbcr2rgb(img)

    Qmtx = nsqjpeg.Qmtx
    Q_1 = nsqjpeg.q_ops_obj.Q_list[0]
    Q_2 = nsqjpeg.q_ops_obj.Q_list[2]
    Q_3 = nsqjpeg.q_ops_obj.Q_list[-1]

    q_1_br = nsqjpeg.q_ops_obj.rate_list[0]/(true_N[0]*true_N[1])
    q_2_br = nsqjpeg.q_ops_obj.rate_list[2]/(true_N[0]*true_N[1])
    q_3_br = nsqjpeg.q_ops_obj.rate_list[-1]/(true_N[0]*true_N[1])

    n_cwd_1 = nsqjpeg.q_ops_obj.n_cwds[0]
    n_cwd_2 = nsqjpeg.q_ops_obj.n_cwds[2]
    n_cwd_3 = nsqjpeg.q_ops_obj.n_cwds[-1]

    plt.figure(figsize=(25, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(img_show/255)
    plt.title('Original', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 5, 2)
    im = plt.imshow(Qmtx, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('IDQD matrix', fontsize=16)
    plt.colorbar(im)
    plt.subplot(1, 5, 3)
    im = plt.imshow(Q_1, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('CW: '+str(n_cwd_1) + '. OH: ' + str(np.round(q_1_br, 3)) + ' bpp', fontsize=16)
    plt.colorbar(im)
    plt.subplot(1, 5, 4)
    im = plt.imshow(Q_2, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('CW: '+str(n_cwd_2) + '. OH: ' + str(np.round(q_2_br, 3)) + ' bpp', fontsize=16)
    plt.colorbar(im)
    plt.subplot(1, 5, 5)
    im = plt.imshow(Q_3, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('CW: '+str(n_cwd_3) + '. OH: ' + str(np.round(q_3_br, 3)) + ' bpp', fontsize=16)
    plt.colorbar(im)

    # set them to tight
    plt.tight_layout()
    plt.show()

