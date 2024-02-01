from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect, ycbcr2rgb, rgb2ycbcr   
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim as compute_Q_class
from handling_q.q_ops import q_ops_ssim as compute_ops_class

import random
from iqa_funs.brisque_torch import brisque
import torch
from utils.q_utils import compute_LPIPS_gs
from utils.q_utils import ssim_func as ssim_mod
from utils.q_utils import ms_ssim_func as ms_ssim_mod

from utils.bits_class import compute_bits_block

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
flag_uniform = True

jpeg = JPEG(nqs, uniform=flag_uniform, N=8)
compute_Q_obj = compute_Q_class(true_N, sampling_depth=16)
q_ops_obj = compute_ops_class(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
nsqjpeg = NSQJPEG(compute_Q_obj, q_ops_obj, nqs=nqs, N=N, uniform=flag_uniform)


def compress_JPEG(qual_lev, img):
    trans, bits = jpeg.compress_420(img, qual_lev)
    trans = trans[0]
    bits_map = compute_bits_block(trans)
    return bits_map


def compress_NSQJPEG(qual_lev, img):
    trans, bits = nsqjpeg.compress_420(img, qual_lev)
    trans = trans[0]
    bits_map = compute_bits_block(trans)
    return bits_map


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    likelihoods = out_net['likelihoods']['y']
    output = (torch.log(likelihoods).sum() / (-np.log(2) * num_pixels)).item()
    return output


def evaluate_metrics(img1, img2):
    
    if (len(img1.shape) > 2):
        img1 = img1[:, :, 0]
        img2 = img2[:, :, 0]

    if (np.max(np.abs(img1))>1):
        img1 = 2*img1/255-1
    if (np.max(np.abs(img2))>1):
        img2 = 2*img2/255-1
    # compute MSE and LPIPS for each block
    mse = 10*np.log10(255**2/np.mean(np.square(255*(img1 - img2)/2)))
    ten_img1 = torch.tensor(img1+1).unsqueeze(0).unsqueeze(0).type(torch.float32)
    ten_img2 = torch.tensor(img2+1).unsqueeze(0).unsqueeze(0).type(torch.float32)
    lpips = 1-compute_LPIPS_gs(ten_img1-1, ten_img2-1)

    ssim_val = ssim_func(ten_img1, ten_img2)
    msssim_val = ms_ssim_func(ten_img1, ten_img2)
    
    bris_2 = brisque(ten_img2, data_range=2)
    bris_1 = torch.abs(brisque(ten_img1, data_range=2))
    bris = 1 - (bris_2) / (bris_2 + 2*bris_1)
    #bris = -np.log(brisque(ten_img1, data_range=2) / (brisque(ten_img2, data_range=2) + brisque(ten_img1, data_range=2)))
    bris = bris.item()
    return mse, lpips, ssim_val, msssim_val, bris


def get_touples(mag, bits):
    arr_out = np.zeros((len(mag), 2))
    order = np.argsort(bits)
    arr_out[:, 0] = bits[order]
    arr_out[:, 1] = mag[order]
    return arr_out


def get_mean_format(data):
    mean = np.round(np.mean(data), 2)
    return '{}'.format(mean)


path = 'Images/CLIC/Testing/'
dirs = os.listdir(path)
num_images = 20
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]


bdrate_ssim_nsqjpeg = np.zeros(num_images)
bdrate_lpips_nsqjpeg = np.zeros_like(bdrate_ssim_nsqjpeg)
bdrate_mse_nsqjpeg = np.zeros_like(bdrate_lpips_nsqjpeg)
bdrate_msssim_nsqjpeg = np.zeros_like(bdrate_lpips_nsqjpeg)
bdrate_bris_nsqjpeg = np.zeros_like(bdrate_lpips_nsqjpeg)

bdrate_ssim_bmshj2018_factorized = np.zeros_like(bdrate_ssim_nsqjpeg)
bdrate_lpips_bmshj2018_factorized = np.zeros_like(bdrate_ssim_nsqjpeg)
bdrate_mse_bmshj2018_factorized = np.zeros_like(bdrate_ssim_nsqjpeg)
bdrate_msssim_bmshj2018_factorized = np.zeros_like(bdrate_ssim_nsqjpeg)
bdrate_bris_bmshj2018_factorized = np.zeros_like(bdrate_ssim_nsqjpeg)

bdrate_ssim_cheng2020_anchor = np.zeros_like(bdrate_ssim_nsqjpeg)
bdrate_lpips_cheng2020_anchor = np.zeros_like(bdrate_ssim_nsqjpeg)
bdrate_mse_cheng2020_anchor = np.zeros_like(bdrate_ssim_nsqjpeg)
bdrate_msssim_cheng2020_anchor = np.zeros_like(bdrate_ssim_nsqjpeg)
bdrate_bris_cheng2020_anchor = np.zeros_like(bdrate_ssim_nsqjpeg)


mse_vals = np.zeros((nqs, len(dirs)))
mse_vals_nsqjpeg = np.zeros_like(mse_vals)
mse_vals_bmshj2018_factorized = np.zeros_like(mse_vals)
mse_vals_cheng2020_anchor = np.zeros_like(mse_vals)

lpips_vals = np.zeros_like(mse_vals)
lpips_vals_nsqjpeg = np.zeros_like(mse_vals)
lpips_vals_bmshj2018_factorized = np.zeros_like(mse_vals)
lpips_vals_cheng2020_anchor = np.zeros_like(mse_vals)

ssim_vals = np.zeros_like(mse_vals)
ssim_vals_nsqjpeg = np.zeros_like(mse_vals)
ssim_vals_bmshj2018_factorized = np.zeros_like(mse_vals)
ssim_vals_cheng2020_anchor = np.zeros_like(mse_vals)

msssim_vals = np.zeros_like(mse_vals)
msssim_vals_nsqjpeg = np.zeros_like(mse_vals)
msssim_vals_bmshj2018_factorized = np.zeros_like(mse_vals)
msssim_vals_cheng2020_anchor = np.zeros_like(mse_vals)

bris_vals = np.zeros_like(mse_vals)
bris_vals_nsqjpeg = np.zeros_like(mse_vals)
bris_vals_bmshj2018_factorized = np.zeros_like(mse_vals)
bris_vals_cheng2020_anchor = np.zeros_like(mse_vals)

# create array to store bits
bits = []
bits_savc = []
bits_qsavc = []
bits_nsqjpeg = []
bits_bmshj2018_factorized = []
bits_chen2020_anchor = []

for i in range(num_images):
    if (i != 19):
        continue
    fullpath = os.path.join(path,dirs[i])  
    img, depth = read_image_resize_rect(fullpath, true_N)
    #img = img[:, :, 0]
    #depth = 1

    jpeg.set_Q(img)
    nsqjpeg.set_Q(img, one_depth=True)
    nsqjpeg.set_basis()

    bits_img = []
    bits_img_savc = []
    bits_img_nsqjpeg = []
    bits_img_qsavc = []
    bits_img_bmshj2018_factorized = []
    bits_img_cheng2020_anchor = []

    img_show = ycbcr2rgb(img)
    Qmtx = nsqjpeg.Qmtx

    for j in range(nqs):

        qual_idx = j
        map_jpeg = compress_JPEG(j, img)
        map_nsqjpeg = compress_NSQJPEG(j, img)

        map_jpeg_im = np.zeros((map_jpeg.shape[0]*N, map_jpeg.shape[1]*N))
        map_nsqjpeg_im = np.zeros_like(map_jpeg_im)
        for k in range(map_jpeg.shape[0]):
            for l in range(map_jpeg.shape[1]):
                map_jpeg_im[k*N:(k+1)*N, l*N:(l+1)*N] = map_jpeg[k, l]
                map_nsqjpeg_im[k*N:(k+1)*N, l*N:(l+1)*N] = map_nsqjpeg[k, l]

        plt.figure(figsize=(20, 20))
        plt.subplot(2, 2, 1)
        plt.imshow(img_show/255)
        plt.title('Original', fontsize=16)
        plt.xticks([])
        plt.yticks([])  
        plt.subplot(2, 2, 2)
        im = plt.imshow(map_jpeg_im, cmap='gray')
        plt.title('JPEG bit map', fontsize=16)
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(im)
        plt.subplot(2, 2, 3)
        plt.imshow(Qmtx)
        im = plt.imshow(Qmtx, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Qmtx', fontsize=16)
        plt.colorbar(im)
        plt.subplot(2, 2, 4)
        im = plt.imshow(map_nsqjpeg_im, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('NSQJPEG bit map', fontsize=16)
        plt.colorbar(im)
        plt.show()


