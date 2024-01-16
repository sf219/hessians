from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_pytorch import compute_Q_msssim as compute_Q_class

import random
from iqa_funs.brisque_torch import brisque
import torch
from utils.q_utils import ssim_func, ms_ssim_func, compute_LPIPS_gs

n_cwd = 4
true_N = (256, 256)
nqs = 6
N = 8

###### REMEMBER THAT NSQJPEG IS MODIFIED TO NOT INCLUDE RLGR ENCODING ######
###### 
flag_uniform = True

jpeg = JPEG(nqs, uniform=flag_uniform, N=8)
compute_Q_obj = compute_Q_class(true_N=true_N, n_cwd=n_cwd, N=8)
nsqjpeg = NSQJPEG(compute_Q_obj, nqs=nqs, N=N, uniform=flag_uniform)


def compress_JPEG(qual_lev, img):
    trans, bits = jpeg.compress(img, depth, qual_lev)
    comp_img = jpeg.uncompress(trans, depth, qual_lev)
    return comp_img, bits


def compress_NSQJPEG(qual_lev, img):
    trans, bits = nsqjpeg.compress(img, depth, qual_lev)
    comp_img = nsqjpeg.uncompress(trans, depth, qual_lev)
    return comp_img, bits


def evaluate_metrics(img1, img2):
    if (np.max(np.abs(img1))>1):
        img1 = 2*img1/255-1
    if (np.max(np.abs(img2))>1):
        img2 = 2*img2/255-1
    # compute MSE and LPIPS for each block
    mse = 10*np.log10(255**2/np.mean(np.square(255*(img1 - img2)/2)))
    ten_img1 = torch.tensor(img1+1).unsqueeze(0).unsqueeze(0).type(torch.float32)
    ten_img2 = torch.tensor(img2+1).unsqueeze(0).unsqueeze(0).type(torch.float32)
    lpips = 1-compute_LPIPS_gs(ten_img1-1, ten_img2-1)

    ssim_val = ssim_func(ten_img1, ten_img2).item()
    msssim_val = ms_ssim_func(ten_img1, ten_img2).item()
    
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

mse_vals = np.zeros((nqs, len(dirs)))
mse_vals_nsqjpeg = np.zeros_like(mse_vals)

lpips_vals = np.zeros_like(mse_vals)
lpips_vals_nsqjpeg = np.zeros_like(mse_vals)

ssim_vals = np.zeros_like(mse_vals)
ssim_vals_nsqjpeg = np.zeros_like(mse_vals)

msssim_vals = np.zeros_like(mse_vals)
msssim_vals_nsqjpeg = np.zeros_like(mse_vals)

bris_vals = np.zeros_like(mse_vals)
bris_vals_nsqjpeg = np.zeros_like(mse_vals)

# create array to store bits
bits = []
bits_savc = []
bits_qsavc = []
bits_nsqjpeg = []

for i in range(num_images):
    fullpath = os.path.join(path,dirs[i])  
    img, depth = read_image_resize_rect(fullpath, true_N)
    img = img[:, :, 0]
    depth = 1

    jpeg.set_Q(img)
    nsqjpeg.set_Q(img)

    bits_img = []
    bits_img_savc = []
    bits_img_nsqjpeg = []
    bits_img_qsavc = []

    for j in range(nqs):
        qual_idx = j
        comp_img_jpeg, bits_tmp = compress_JPEG(j, img)
        comp_img_nsqjpeg, bits_nsqjpeg_tmp = compress_NSQJPEG(j, img)

        bits_img.append(bits_tmp)
        bits_img_nsqjpeg.append(bits_nsqjpeg_tmp)

        mse_vals[j, i], lpips_vals[j, i], ssim_vals[j, i], msssim_vals[j, i], bris_vals[j, i] = evaluate_metrics(img, comp_img_jpeg)
        mse_vals_nsqjpeg[j, i], lpips_vals_nsqjpeg[j, i], ssim_vals_nsqjpeg[j, i], msssim_vals_nsqjpeg[j, i], bris_vals_nsqjpeg[j, i] = evaluate_metrics(img, comp_img_nsqjpeg)

    bits.append(bits_img)
    bits_nsqjpeg.append(bits_img_nsqjpeg)

    total_bits = np.array([bits_img[j].bits_over for j in range(nqs)])/img.size
    total_bits_nsqjpeg = np.array([bits_img_nsqjpeg[j].bits_over for j in range(nqs)])/img.size
    
    Qmtx = nsqjpeg.Qmtx
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(total_bits, mse_vals[:, i], label='JPEG')
    plt.plot(total_bits_nsqjpeg, mse_vals_nsqjpeg[:, i], label='NSQJPEG')
    plt.legend()
    plt.title('MSE')
    plt.subplot(2, 2, 2)
    plt.plot(total_bits, msssim_vals[:, i], label='JPEG')
    plt.plot(total_bits_nsqjpeg, msssim_vals_nsqjpeg[:, i], label='NSQJPEG')
    plt.legend()
    plt.title('MSSSIM')
    plt.subplot(2, 2, 3)
    im = plt.imshow(img, cmap='gray')
    plt.title('Original')
    plt.colorbar(im)
    plt.subplot(2, 2, 4)
    im = plt.imshow(Qmtx, cmap='gray')
    plt.title('Qmtx')
    plt.colorbar(im)
    plt.tight_layout()
    str_out = 'sketch/simul_tmps/{}.pdf'.format(i)
    plt.savefig(str_out, bbox_inches='tight')    
    #plt.show()

    touple_mse_jpeg = get_touples(mse_vals[:, i], total_bits)
    touple_mse_nsqjpeg = get_touples(mse_vals_nsqjpeg[:, i], total_bits_nsqjpeg)

    touple_lpips_jpeg = get_touples(lpips_vals[:, i], total_bits)
    touple_lpips_nsqjpeg = get_touples(lpips_vals_nsqjpeg[:, i], total_bits_nsqjpeg)

    touple_ssim_jpeg = get_touples(ssim_vals[:, i], total_bits)
    touple_ssim_nsqjpeg = get_touples(ssim_vals_nsqjpeg[:, i], total_bits_nsqjpeg)

    touple_msssim_jpeg = get_touples(msssim_vals[:, i], total_bits)
    touple_msssim_nsqjpeg = get_touples(msssim_vals_nsqjpeg[:, i], total_bits_nsqjpeg)

    touple_bris_jpeg = get_touples(bris_vals[:, i], total_bits)
    touple_bris_nsqjpeg = get_touples(bris_vals_nsqjpeg[:, i], total_bits_nsqjpeg)


    bdrate_mse_nsqjpeg[i] = bdrate(touple_mse_jpeg, touple_mse_nsqjpeg)
    bdrate_lpips_nsqjpeg[i] = bdrate(touple_lpips_jpeg, touple_lpips_nsqjpeg)
    bdrate_ssim_nsqjpeg[i] = bdrate(touple_ssim_jpeg, touple_ssim_nsqjpeg)
    bdrate_msssim_nsqjpeg[i] = bdrate(touple_msssim_jpeg, touple_msssim_nsqjpeg)

    bdrate_bris_nsqjpeg[i] = bdrate(touple_bris_jpeg, touple_bris_nsqjpeg)

    nsqjpeg_mse = get_mean_format(bdrate_mse_nsqjpeg[i])
    nsqjpeg_lpips = get_mean_format(bdrate_lpips_nsqjpeg[i])
    nsqjpeg_ssim = get_mean_format(bdrate_ssim_nsqjpeg[i])
    nsqjpeg_msssim = get_mean_format(bdrate_msssim_nsqjpeg[i])
    nsqjpeg_bris = get_mean_format(bdrate_bris_nsqjpeg[i])

    print('nsqjpeg image ' + str(i+1) + ' MSE: ' + nsqjpeg_mse + ' SSIM: ' + nsqjpeg_ssim + ' LPIPS: ' + nsqjpeg_lpips + ' MSSSIM: ' + nsqjpeg_msssim + ' BRIS: ' + nsqjpeg_bris)

    nsqjpeg_mse = get_mean_format(bdrate_mse_nsqjpeg[:i+1])
    nsqjpeg_lpips = get_mean_format(bdrate_lpips_nsqjpeg[:i+1])
    nsqjpeg_ssim = get_mean_format(bdrate_ssim_nsqjpeg[:i+1])
    nsqjpeg_msssim = get_mean_format(bdrate_msssim_nsqjpeg[:i+1])
    nsqjpeg_bris = get_mean_format(bdrate_bris_nsqjpeg[:i+1])
    print('Mean nsqjpeg MSE: ' + nsqjpeg_mse + ' SSIM: ' + nsqjpeg_ssim + ' LPIPS: ' + nsqjpeg_lpips + ' MSSSIM: ' + nsqjpeg_msssim + ' BRIS: ' + nsqjpeg_bris)