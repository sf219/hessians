from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect, ycbcr2rgb
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim as compute_Q_class
from handling_q.q_ops import q_ops_ssim as compute_ops_class
from handling_q.q_ops_noco import q_ops_ssim as compute_ops_class_noco
import random
from utils.q_utils_jax import compute_LPIPS_color
from iqa_funs.jax_ssim import jax_ssim_eval as ssim_mod
from iqa_funs.jax_ssim import jax_msssim as ms_ssim_mod
import jax.numpy as jnp

from skimage.metrics import structural_similarity as ssim

ssim_func = lambda x, y: -10*np.log10(ssim_mod(x, y))
ms_ssim_func = lambda x, y: -10*np.log10(ms_ssim_mod(x, y))

#ssim_func = lambda x, y: ssim_mod(x, y)
#ms_ssim_func = lambda x, y: ms_ssim_mod(x, y)


n_cwd = 4
true_N = (512, 512)
nqs = 8
N = 8

###### REMEMBER THAT NSQJPEG IS MODIFIED TO NOT INCLUDE RLGR ENCODING ######
###### 
flag_uniform = True

jpeg = JPEG(nqs, uniform=flag_uniform, N=8)
compute_Q_obj = compute_Q_class(true_N, sampling_depth=16)
q_ops_obj = compute_ops_class(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
q_ops_obj_noco = compute_ops_class_noco(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
nsqjpeg = NSQJPEG(compute_Q_obj, q_ops_obj, nqs=nqs, N=N, uniform=flag_uniform, quant_scal=65)


def compress_JPEG(qual_lev, img):
    trans, bits, bits_arr = jpeg.compress_420(img, qual_lev)
    comp_img = jpeg.uncompress_420(trans, qual_lev)
    return comp_img, bits, bits_arr


def compress_NSQJPEG(qual_lev, img):
    trans, bits, bits_arr = nsqjpeg.compress_420(img, qual_lev)
    comp_img = nsqjpeg.uncompress_420(trans, qual_lev)
    return comp_img, bits, bits_arr


def evaluate_metrics(img1, img2):
    weights = np.array([8, 0, 0])
    weights = weights/np.sum(weights)
    mse_score = np.zeros((3))
    ssim_score = np.zeros((3))
    msssim_score = np.zeros((3))

    # first compute the LPIPS for the full image
    img_1_rgb = ycbcr2rgb(img1)
    img_2_rgb = ycbcr2rgb(img2)

    img_1_rgb = 2*img_1_rgb/255-1
    img_2_rgb = 2*img_2_rgb/255-1

    ten_img_1_rgb = jnp.array(img_1_rgb)
    ten_img_2_rgb = jnp.array(img_2_rgb)

    lpips = -10*np.log10(compute_LPIPS_color(ten_img_1_rgb, ten_img_2_rgb))

    # now compute the other metrics for each channel
    img1 = 2*img1/255
    img2 = 2*img2/255

    for i in range(3):
        img_1_tmp = img1[:, :, i] 
        img_2_tmp = img2[:, :, i]
        mse_score[i] = np.mean(np.square(255*(img_1_tmp - img_2_tmp)/2))
        ten_img1 = jnp.array(img_1_tmp)
        ten_img2 = jnp.array(img_2_tmp)
        ssim_score[i] = ssim_func(ten_img1, ten_img2)
        #ssim_score[i] = ssim(img_1_tmp, img_2_tmp, data_range=2)
        msssim_score[i] = ms_ssim_func(ten_img1, ten_img2)
    ssim_score[np.isnan(ssim_score)] = 0
    msssim_score[np.isnan(msssim_score)] = 0
    ssim_score[np.isinf(ssim_score)] = 0
    msssim_score[np.isinf(msssim_score)] = 0
    ssim_val = ssim_score @ weights
    msssim_val = msssim_score @ weights
    mse_val = mse_score @ weights
    psnr = 10*np.log10(255**2/mse_val)
    return psnr, lpips, ssim_val, msssim_val


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

    for j in range(nqs):

        qual_idx = j
        comp_img_jpeg, bits_tmp, bits_arr_1 = compress_JPEG(j, img)
        comp_img_nsqjpeg, bits_nsqjpeg_tmp, bits_arr_2 = compress_NSQJPEG(j, img)

        bits_tmp = np.sum(bits_arr_1)
        bits_nsqjpeg_tmp = np.sum(bits_arr_2)        

        bits_img.append(bits_tmp)
        bits_img_nsqjpeg.append(bits_nsqjpeg_tmp)

        mse_vals[j, i], lpips_vals[j, i], ssim_vals[j, i], msssim_vals[j, i] = evaluate_metrics(img, comp_img_jpeg)
        mse_vals_nsqjpeg[j, i], lpips_vals_nsqjpeg[j, i], ssim_vals_nsqjpeg[j, i], msssim_vals_nsqjpeg[j, i] = evaluate_metrics(img, comp_img_nsqjpeg)

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
