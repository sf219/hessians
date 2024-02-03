from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim as compute_Q_class
from handling_q.q_ops import q_ops_ssim as compute_ops_class

import random
import torch
from utils.q_utils import ssim_func as ssim_mod
from utils.q_utils import ms_ssim_func as ms_ssim_mod
from scipy.spatial import ConvexHull

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

jpeg = JPEG(nqs, uniform=flag_uniform, N=8)
compute_Q_obj = compute_Q_class(true_N, sampling_depth=16)
q_ops_obj = compute_ops_class(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
nsqjpeg = NSQJPEG(compute_Q_obj, q_ops_obj, nqs=nqs, N=N, uniform=flag_uniform)


def compress_JPEG(qual_lev, img):
    trans, bits = jpeg.compress_420(img, qual_lev)
    comp_img = jpeg.uncompress_420(trans, qual_lev)
    return comp_img, bits


def compress_NSQJPEG(qual_lev, img):
    trans, bits = nsqjpeg.compress_420(img, qual_lev)
    comp_img = nsqjpeg.uncompress_420(trans, qual_lev)
    return comp_img, bits


path = 'Images/CLIC/Testing/'
dirs = os.listdir(path)
num_images = 40
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

for i in range(num_images):
    fullpath = os.path.join(path, dirs[i])
    img, depth = read_image_resize_rect(fullpath, true_N)
    #img = img[:, :, 0]
    #depth = 1

    jpeg.set_Q(img)
    nsqjpeg.set_Q(img, one_depth=True)

    rate_list = nsqjpeg.q_ops_obj.rate_list
    SSE_list = nsqjpeg.q_ops_obj.SSE_list

    pairs = list(zip(rate_list, SSE_list))
    cv_hull = ConvexHull(pairs)
    vertices = np.sort(cv_hull.vertices)

    deg = 4
    fit = np.polyfit(rate_list, SSE_list, deg=deg)
    der = fit[0:-1]*np.arange(deg, 0, -1)
    p = np.poly1d(fit)
    der_p = np.poly1d(der)

    der_der = der[0:-1]*np.arange(deg-1, 0, -1)
    der_p_p = np.poly1d(der_der)

    plt.figure()
    plt.plot(rate_list, SSE_list, 'o', label='Original data')
    x_axis = np.linspace(min(rate_list), max(rate_list), 100)
    plt.plot(x_axis, der_p(x_axis), label='Fit der')
    plt.plot(x_axis, der_p_p(x_axis), label='Fit der der')
    plt.plot(x_axis, p(x_axis), label='Fit')
    plt.xlabel('Rate', fontsize=16)
    plt.ylabel('SSE', fontsize=16)
    plt.legend(fontsize=16)
    plt.title('Rate vs SSE', fontsize=16)

    plt.figure()
    plt.plot(rate_list, SSE_list, 'o', label='Original data')
    x_axis = np.linspace(min(rate_list), max(rate_list), 100)
    plt.plot(x_axis, p(x_axis), label='Fit')
    plt.xlabel('Rate', fontsize=16)
    plt.ylabel('SSE', fontsize=16)
    plt.legend(fontsize=16)
    plt.title('Rate vs SSE', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    plt.figure()
    plt.plot(rate_list, 10*np.log10(SSE_list), 'o', label='Original data')
    x_axis = np.linspace(min(rate_list), max(rate_list), 100)    
    plt.plot(np.array(rate_list)[vertices], 10*np.log10(np.array(SSE_list)[vertices]), '--', lw=2, label='Convex Hull')
    plt.xlabel('Rate', fontsize=16)
    plt.ylabel('SSE', fontsize=16)
    plt.legend(fontsize=16)
    plt.title('Rate vs SSE', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()