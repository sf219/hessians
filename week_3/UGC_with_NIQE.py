from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_brisque as compute_Q_class_brisque
from compute_q.compute_Q_jax import compute_Q_niqe as compute_Q_class_niqe

from handling_q.q_ops import q_ops_brisque as compute_ops_class_brisque
from handling_q.q_ops import q_ops_niqe as compute_ops_class_niqe

import random
from iqa_funs.brisque_torch import brisque
from iqa_funs.niqe_jax import niqe_loss

import torch
from utils.utils_lpit import ycbcr2rgb
import jax.numpy as jnp

n_cwd = 4
true_N = (480, 480)
nqs = 6
N = 8


###### REMEMBER THAT NSQJPEG IS MODIFIED TO NOT INCLUDE RLGR ENCODING ######
###### 
flag_uniform = True

jpeg = JPEG(nqs, uniform=flag_uniform, N=8)
compute_Q_obj = compute_Q_class_brisque(true_N=true_N, sampling_depth=16)
q_ops_obj = compute_ops_class_brisque(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
nsqjpeg_bris = NSQJPEG(compute_Q_obj, q_ops_obj, nqs=nqs, N=N, uniform=flag_uniform)

compute_Q_obj = compute_Q_class_niqe(true_N=true_N, sampling_depth=16)
q_ops_obj = compute_ops_class_niqe(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
nsqjpeg_niqe = NSQJPEG(compute_Q_obj, q_ops_obj, nqs=nqs, N=N, uniform=flag_uniform)


def compress_JPEG(qual_lev, img):
    trans, bits = jpeg.compress_420(img, qual_lev)
    comp_img = jpeg.uncompress_420(trans, qual_lev)
    return comp_img, bits


def compress_NSQJPEG_bris(qual_lev, img):
    trans, bits = nsqjpeg_bris.compress_420(img, qual_lev)
    comp_img = nsqjpeg_bris.uncompress_420(trans, qual_lev)
    return comp_img, bits


def compress_NSQJPEG_niqe(qual_lev, img):
    trans, bits = nsqjpeg_niqe.compress_420(img, qual_lev)
    comp_img = nsqjpeg_niqe.uncompress_420(trans, qual_lev)
    return comp_img, bits


def compute_bare_brisque(img):
    img = img[:, :, 0]
    if (np.max(np.abs(img))>1):
        img = 2*img/255-1
    ten_img = torch.tensor(img+1).unsqueeze(0).unsqueeze(0).type(torch.float32)
    bris = brisque(ten_img, data_range=2)
    return bris.item()

def compute_bare_niqe(img):
    img = img[:, :, 0]
    if (np.max(np.abs(img))>1):
        img = 2*img/255
    ten_img = jnp.array(img.squeeze())
    niqe = niqe_loss(ten_img)
    return np.array(niqe)

def get_touples(mag, bits):
    arr_out = np.zeros((len(mag), 2))
    order = np.argsort(bits)
    arr_out[:, 0] = bits[order]
    arr_out[:, 1] = mag[order]
    return arr_out


def get_mean_format(data):
    mean = np.round(np.mean(data), 2)
    return '{}'.format(mean)


path = 'Images/UGC_examples/Testing/'
dirs = os.listdir(path)
num_images = 10
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

mse_vals = np.zeros((nqs, len(dirs)))
mse_vals_nsqjpeg = np.zeros_like(mse_vals)


true_bris_vals = np.zeros_like(mse_vals)
true_bris_vals_nsqjpeg_bris = np.zeros_like(mse_vals)
true_bris_vals_nsqjpeg_niqe = np.zeros_like(mse_vals)

true_niqe_vals = np.zeros_like(mse_vals)
true_niqe_vals_nsqjpeg_bris = np.zeros_like(mse_vals)
true_niqe_vals_nsqjpeg_niqe = np.zeros_like(mse_vals)

# create array to store bits
bits = []
bits_nsqjpeg = []

for i in range(num_images):
    fullpath = os.path.join(path,dirs[i])  
    img, depth = read_image_resize_rect(fullpath, true_N)
    # replace UGC by PBD

    img_show = ycbcr2rgb(img)

    jpeg.set_Q(img)
    nsqjpeg_bris.set_Q(img, one_depth=True)
    nsqjpeg_bris.set_basis()
    nsqjpeg_niqe.set_Q(img, one_depth=True)
    nsqjpeg_niqe.set_basis()

    bits_img = []
    bits_img_nsqjpeg_bris = []
    bits_img_nsqjpeg_niqe = []

    for j in range(nqs):
        qual_idx = j
        comp_img_jpeg, bits_tmp = compress_JPEG(j, img)
        comp_img_nsqjpeg_bris, bits_nsqjpeg__bris_tmp = compress_NSQJPEG_bris(j, img)
        comp_img_nsqjpeg_niqe, bits_nsqjpeg_niqe_tmp = compress_NSQJPEG_niqe(j, img)

        bits_img.append(bits_tmp)
        bits_img_nsqjpeg_bris.append(bits_nsqjpeg__bris_tmp)
        bits_img_nsqjpeg_niqe.append(bits_nsqjpeg_niqe_tmp)

        true_bris_vals[j, i] = compute_bare_brisque(comp_img_jpeg)
        true_bris_vals_nsqjpeg_bris[j, i] = compute_bare_brisque(comp_img_nsqjpeg_bris)
        true_bris_vals_nsqjpeg_niqe[j, i] = compute_bare_brisque(comp_img_nsqjpeg_niqe)

        true_niqe_vals[j, i] = compute_bare_niqe(comp_img_jpeg)
        true_niqe_vals_nsqjpeg_bris[j, i] = compute_bare_niqe(comp_img_nsqjpeg_bris)
        true_niqe_vals_nsqjpeg_niqe[j, i] = compute_bare_niqe(comp_img_nsqjpeg_niqe)

    total_bits = np.array([bits_img[j] for j in range(nqs)])/img.size
    total_bits_nsqjpeg_bris = np.array([bits_img_nsqjpeg_bris[j] for j in range(nqs)])/img.size
    total_bits_nsqjpeg_niqe = np.array([bits_img_nsqjpeg_niqe[j] for j in range(nqs)])/img.size

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_show)
    plt.title('Original image', fontsize=18)
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(nsqjpeg_bris.Qmtx, cmap='gray')
    plt.title('BRISQUE Matrix', fontsize=18)
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(nsqjpeg_niqe.Qmtx, cmap='gray')
    plt.title('NIQE Matrix', fontsize=18)
    plt.axis('off')

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(total_bits, true_bris_vals[:, i], 'o-', label='JPEG')
    plt.plot(total_bits_nsqjpeg_bris, true_bris_vals_nsqjpeg_bris[:, i], 'o-', label='BRISQUE-OPT')
    plt.plot(total_bits_nsqjpeg_niqe, true_bris_vals_nsqjpeg_niqe[:, i], 'o-', label='NIQE-OPT')
    plt.title('BRISQUE', fontsize=18)
    plt.xlabel('Bits', fontsize=18)
    plt.ylabel('BRISQUE', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    plt.subplot(1, 2, 2)
    plt.plot(total_bits, true_niqe_vals[:, i], 'o-', label='JPEG')
    plt.plot(total_bits_nsqjpeg_bris, true_niqe_vals_nsqjpeg_bris[:, i], 'o-', label='BRISQUE-OPT')
    plt.plot(total_bits_nsqjpeg_niqe, true_niqe_vals_nsqjpeg_niqe[:, i], 'o-', label='NIQE-OPT')
    plt.title('NIQE', fontsize=18)
    plt.xlabel('Bits', fontsize=18)
    plt.ylabel('NIQE', fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(fontsize=18)
    # set axis tight
    plt.tight_layout()
    str_out = 'sketch/simul_tmps/{}_new_ugc_rd.pdf'.format(i)
    plt.savefig(str_out, bbox_inches='tight')    
    plt.show()
