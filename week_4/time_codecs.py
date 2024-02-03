from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_msssim as compute_Q_class
from handling_q.q_ops import q_ops_msssim as compute_ops_class

import random
import torch
from utils.q_utils import ssim_func as ssim_mod
from utils.q_utils import ms_ssim_func as ms_ssim_mod

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


def run_test():
    path = 'Images/CLIC/Testing/'
    dirs = os.listdir(path)
    num_images = 1
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
        for j in range(nqs):
            qual_idx = j
            comp_img_jpeg, bits_tmp = compress_JPEG(j, img)
            comp_img_nsqjpeg, bits_nsqjpeg_tmp = compress_NSQJPEG(j, img)


import cProfile, pstats, io
from pstats import SortKey

pr = cProfile.Profile()
pr.enable()
run_test()
pr.disable()
s = io.StringIO()
sortby = SortKey.CUMULATIVE
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.reverse_order().print_stats()
print(s.getvalue())
breakpoint()