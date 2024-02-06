from utils.utils_lpit import bdrate
from my_codecs.JPEG_jax import JPEG
from my_codecs.NSQJPEG_jax import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect, ycbcr2rgb 
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim as compute_Q_class
from handling_q.q_ops import q_ops_ssim as compute_ops_class

import random

#ssim_func = lambda x, y: -10*np.log10(ssim_mod(x, y))
#ms_ssim_func = lambda x, y: -10*np.log10(ms_ssim_mod(x, y))

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

def get_touples(mag, bits):
    arr_out = np.zeros((len(mag), 2))
    order = np.argsort(bits)
    arr_out[:, 0] = bits[order]
    arr_out[:, 1] = mag[order]
    return arr_out


def get_mean_format(data):
    mean = np.round(np.mean(data), 2)
    return '{}'.format(mean)


path = 'Images/CLIC/Sky/'
dirs = os.listdir(path)
num_images = 40
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

bdrate_ssim_nsqjpeg = np.zeros(num_images)
bdrate_lpips_nsqjpeg = np.zeros_like(bdrate_ssim_nsqjpeg)
bdrate_mse_nsqjpeg = np.zeros_like(bdrate_lpips_nsqjpeg)
bdrate_msssim_nsqjpeg = np.zeros_like(bdrate_lpips_nsqjpeg)

mse_vals = np.zeros((nqs, len(dirs)))
mse_vals_nsqjpeg = np.zeros_like(mse_vals)

lpips_vals = np.zeros_like(mse_vals)
lpips_vals_nsqjpeg = np.zeros_like(mse_vals)

ssim_vals = np.zeros_like(mse_vals)
ssim_vals_nsqjpeg = np.zeros_like(mse_vals)

msssim_vals = np.zeros_like(mse_vals)
msssim_vals_nsqjpeg = np.zeros_like(mse_vals)

# create array to store bits
bits = []
bits_nsqjpeg = []

for i in range(num_images):
    fullpath = os.path.join(path,dirs[i])  
    img, depth = read_image_resize_rect(fullpath, true_N)
    #img = img[:, :, 0]
    #depth = 1

    jpeg.set_Q(img)
    nsqjpeg.set_Q(img, one_depth=True)

    bits_img = []
    bits_img_savc = []
    bits_img_nsqjpeg = []
    bits_img_qsavc = []

    plt.figure(figsize=(35, 10))
    plt.subplot(2, nqs+1, 1)
    img_show = ycbcr2rgb(img)
    im = plt.imshow(img_show/255)
    plt.title('Original', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, nqs+1, nqs+2)
    Qmtx = nsqjpeg.Qmtx
    im = plt.imshow(Qmtx, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('Qmtx', fontsize=16)
    plt.colorbar(im)
    for j in range(nqs):
        qual_idx = j
        comp_img_jpeg, bits_tmp = compress_JPEG(j, img)
        comp_img_nsqjpeg, bits_nsqjpeg_tmp = compress_NSQJPEG(j, img)

        bits_img.append(bits_tmp)
        bits_img_nsqjpeg.append(bits_nsqjpeg_tmp)

        plt.subplot(2, nqs+1, (j+1)+1)
        img_show = ycbcr2rgb(comp_img_jpeg)
        im = plt.imshow(img_show/255)
        plt.title('JPEG Q={}'.format(j))
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2, nqs+1, (j + 1) + nqs + 2)
        img_show = ycbcr2rgb(comp_img_nsqjpeg)
        im = plt.imshow(img_show/255)
        plt.title('NSQJPEG Q={}'.format(j))
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    str_out = 'sketch/simul_tmps/{}_comp.pdf'.format(i)
    plt.savefig(str_out, bbox_inches='tight')

    bits.append(bits_img)
    bits_nsqjpeg.append(bits_img_nsqjpeg)

    total_bits = np.array([bits_img[j] for j in range(nqs)])/(img.shape[0]*img.shape[1])
    total_bits_nsqjpeg = np.array([bits_img_nsqjpeg[j] for j in range(nqs)])/(img.shape[0]*img.shape[1])

    Qmtx = nsqjpeg.Qmtx
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.plot(total_bits, mse_vals[:, i], label='JPEG',linewidth=3)
    plt.plot(total_bits_nsqjpeg, mse_vals_nsqjpeg[:, i], label='NSQJPEG', linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=16)
    plt.title('MSE', fontsize=16)
    plt.subplot(2, 2, 2)
    plt.plot(total_bits, msssim_vals[:, i], label='JPEG', linewidth=3)
    plt.plot(total_bits_nsqjpeg, msssim_vals_nsqjpeg[:, i], label='NSQJPEG', linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 16)
    plt.title('MSSSIM', fontsize=16)
    plt.subplot(2, 2, 3)
    plt.plot(total_bits, ssim_vals[:, i], label='JPEG', linewidth=3)
    plt.plot(total_bits_nsqjpeg, ssim_vals_nsqjpeg[:, i], label='NSQJPEG', linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 16)
    plt.title('SSIM', fontsize=16)
    plt.subplot(2, 2, 4)
    plt.plot(total_bits, lpips_vals[:, i], label='JPEG', linewidth=3)
    plt.plot(total_bits_nsqjpeg, lpips_vals_nsqjpeg[:, i], label='NSQJPEG', linewidth=3)
    plt.legend(fontsize=16)
    plt.title('LPIPS', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
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


    bdrate_mse_nsqjpeg[i] = bdrate(touple_mse_jpeg, touple_mse_nsqjpeg)
    bdrate_lpips_nsqjpeg[i] = bdrate(touple_lpips_jpeg, touple_lpips_nsqjpeg)
    bdrate_ssim_nsqjpeg[i] = bdrate(touple_ssim_jpeg, touple_ssim_nsqjpeg)
    bdrate_msssim_nsqjpeg[i] = bdrate(touple_msssim_jpeg, touple_msssim_nsqjpeg)
    
    nsqjpeg_mse = get_mean_format(bdrate_mse_nsqjpeg[i])
    nsqjpeg_lpips = get_mean_format(bdrate_lpips_nsqjpeg[i])
    nsqjpeg_ssim = get_mean_format(bdrate_ssim_nsqjpeg[i])
    nsqjpeg_msssim = get_mean_format(bdrate_msssim_nsqjpeg[i])

    print('nsqjpeg image ' + str(i+1) + ' MSE: ' + nsqjpeg_mse + ' SSIM: ' + nsqjpeg_ssim + ' LPIPS: ' + nsqjpeg_lpips + ' MSSSIM: ' + nsqjpeg_msssim)

    nsqjpeg_mse = get_mean_format(bdrate_mse_nsqjpeg[:i+1])
    nsqjpeg_lpips = get_mean_format(bdrate_lpips_nsqjpeg[:i+1])
    nsqjpeg_ssim = get_mean_format(bdrate_ssim_nsqjpeg[:i+1])
    nsqjpeg_msssim = get_mean_format(bdrate_msssim_nsqjpeg[:i+1])

    print('Mean nsqjpeg MSE: ' + nsqjpeg_mse + ' SSIM: ' + nsqjpeg_ssim + ' LPIPS: ' + nsqjpeg_lpips + ' MSSSIM: ' + nsqjpeg_msssim)
