from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect, ycbcr2rgb, rgb2ycbcr   
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim as compute_Q_class
from compute_q.compute_Q_jax import compute_Q_ssim_zp as compute_Q_class_zp
from handling_q.q_ops import q_ops_ssim as compute_ops_class

import random
from iqa_funs.brisque_torch import brisque
import torch
from utils.q_utils import compute_LPIPS_gs
from utils.q_utils import ssim_func as ssim_mod
from utils.q_utils import ms_ssim_func as ms_ssim_mod
from iqa_funs.jax_ssim import jax_ssim_zp 

from compressai.zoo import bmshj2018_factorized, cheng2020_anchor
from PIL import Image
from torchvision import transforms
from pytorch_msssim import SSIM, MS_SSIM

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
flag_uniform = False

jpeg = JPEG(nqs, uniform=flag_uniform, N=8)
compute_Q_obj = compute_Q_class(true_N, sampling_depth=16)
compute_Q_obj_zp = compute_Q_class_zp(true_N, sampling_depth=16)



path = 'Images/CLIC/Testing/'
dirs = os.listdir(path)
num_images = 20
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]


for i in range(num_images):
    if i != 1:
        continue
    fullpath = os.path.join(path,dirs[i])  
    img, depth = read_image_resize_rect(fullpath, true_N)

    Q_zp = compute_Q_obj.sample_q(img[:, :, 0].squeeze())
    Q_zp = Q_zp.squeeze()
    Q_zp = Q_zp / np.sum(np.abs(Q_zp)) * Q_zp.size

    Q_sym_pad = Q_zp

    Q_zp = compute_Q_obj_zp.sample_q(img[:, :, 0].squeeze())
    Q_zp = Q_zp.squeeze()
    Q_zp = Q_zp / np.sum(np.abs(Q_zp)) * Q_zp.size

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(Q_sym_pad, cmap='gray')
    plt.colorbar()
    plt.title('Symmetric padding')
    plt.subplot(1, 2, 2)
    plt.imshow(Q_zp, cmap='gray')
    plt.colorbar()
    plt.title('Zero padding')
    plt.show()