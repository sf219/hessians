from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect, ycbcr2rgb, rgb2ycbcr   
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim, compute_Q_msssim, compute_Q_lpips
from handling_q.q_ops import q_ops_ssim, q_ops_msssim, q_ops_lpips

import random
from iqa_funs.brisque_torch import brisque
import torch
from utils.q_utils import compute_LPIPS_gs
from utils.q_utils import ssim_func as ssim_mod
from utils.q_utils import ms_ssim_func as ms_ssim_mod

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
path = 'Images/CLIC/Testing/'
num_images = 40


def run_fun(compute_Q_obj, q_ops_obj, flag_uniform): 
    jpeg = JPEG(nqs, uniform=flag_uniform, N=8)
    nsqjpeg = NSQJPEG(compute_Q_obj, q_ops_obj, nqs=nqs, N=N, uniform=flag_uniform)


    def compress_JPEG(qual_lev, img):
        trans, bits = jpeg.compress_420(img, qual_lev)
        comp_img = jpeg.uncompress_420(trans, qual_lev)
        return comp_img, bits


    def compress_NSQJPEG(qual_lev, img):
        trans, bits = nsqjpeg.compress_420(img, qual_lev)
        comp_img = nsqjpeg.uncompress_420(trans, qual_lev)
        return comp_img, bits


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


    dirs = os.listdir(path)
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
    bits_nsqjpeg = []

    for i in range(num_images):
        fullpath = os.path.join(path,dirs[i])  
        img, depth = read_image_resize_rect(fullpath, true_N)
        #img = img[:, :, 0]
        #depth = 1

        jpeg.set_Q(img)
        nsqjpeg.set_Q(img, one_depth=True)
        nsqjpeg.set_basis()

        bits_img = []
        bits_img_nsqjpeg = []

        for j in range(nqs):

            comp_img_jpeg, bits_tmp = compress_JPEG(j, img)
            comp_img_nsqjpeg, bits_nsqjpeg_tmp = compress_NSQJPEG(j, img)

            bits_img.append(bits_tmp)
            bits_img_nsqjpeg.append(bits_nsqjpeg_tmp)

            mse_vals[j, i], lpips_vals[j, i], ssim_vals[j, i], msssim_vals[j, i], bris_vals[j, i] = evaluate_metrics(img, comp_img_jpeg)
            mse_vals_nsqjpeg[j, i], lpips_vals_nsqjpeg[j, i], ssim_vals_nsqjpeg[j, i], msssim_vals_nsqjpeg[j, i], bris_vals_nsqjpeg[j, i] = evaluate_metrics(img, comp_img_nsqjpeg)


        bits.append(bits_img)
        bits_nsqjpeg.append(bits_img_nsqjpeg)

        total_bits = np.array([bits_img[j] for j in range(nqs)])/(img.shape[0]*img.shape[1])
        total_bits_nsqjpeg = np.array([bits_img_nsqjpeg[j] for j in range(nqs)])/(img.shape[0]*img.shape[1])

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
        print('nsqjpeg image ' + str(i+1) + ' MSE: ' + nsqjpeg_mse + ' SSIM: ' + nsqjpeg_ssim + ' LPIPS: ' + nsqjpeg_lpips + ' MSSSIM: ' + nsqjpeg_msssim + ' BRIS: ' + nsqjpeg_bris, file=open('output.txt', 'a'))

        nsqjpeg_mse = get_mean_format(bdrate_mse_nsqjpeg[:i+1])
        nsqjpeg_lpips = get_mean_format(bdrate_lpips_nsqjpeg[:i+1])
        nsqjpeg_ssim = get_mean_format(bdrate_ssim_nsqjpeg[:i+1])
        nsqjpeg_msssim = get_mean_format(bdrate_msssim_nsqjpeg[:i+1])
        nsqjpeg_bris = get_mean_format(bdrate_bris_nsqjpeg[:i+1])

        print('Mean nsqjpeg MSE: ' + nsqjpeg_mse + ' SSIM: ' + nsqjpeg_ssim + ' LPIPS: ' + nsqjpeg_lpips + ' MSSSIM: ' + nsqjpeg_msssim + ' BRIS: ' + nsqjpeg_bris)
        print('Mean nsqjpeg MSE: ' + nsqjpeg_mse + ' SSIM: ' + nsqjpeg_ssim + ' LPIPS: ' + nsqjpeg_lpips + ' MSSSIM: ' + nsqjpeg_msssim + ' BRIS: ' + nsqjpeg_bris,  file=open('output.txt', 'a'))


flag_uniform = True
compute_Q_obj = compute_Q_lpips(true_N, sampling_depth=16)
q_ops_obj = q_ops_lpips(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
q_ops_obj.n_cwds = [4]

print('LPIPS, uniform')
print('LPIPS, uniform', file=open('output.txt', 'a'))
run_fun(compute_Q_obj, q_ops_obj, flag_uniform)

flag_uniform = False
compute_Q_obj = compute_Q_lpips(true_N, sampling_depth=16)
q_ops_obj = q_ops_lpips(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
q_ops_obj.n_cwds = [4]

print('LPIPS, non-uniform')
print('LPIPS, non-uniform', file=open('output.txt', 'a'))
run_fun(compute_Q_obj, q_ops_obj, flag_uniform)

print('Done')
print('Done', file=open('output.txt', 'a'))