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


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    likelihoods = out_net['likelihoods']['y']
    output = (torch.log(likelihoods).sum() / (-np.log(2) * num_pixels)).item()
    return output


def compress_bmshj2018_factorized(qual_level, img):
    img = ycbcr2rgb(img) #data range should be between [0, 1]
    #img = np.transpose(img, (2, 0, 1))
    # convert to PIL image
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)    
    #breakpoint()
    model = bmshj2018_factorized(quality=qual_level+1, metric='ms-ssim', pretrained=True).eval().to(device)
    model.eval()
    model.update(force=True)   
    #img = torch.tensor(img).unsqueeze(0).type(torch.float64)
    #shape_img = img.shape
    comp_img = model(x)
    #comp_img = model.decompress(comp_img, shape_img).squeeze().numpy()
    bits = compute_bpp(comp_img)
    comp_img = comp_img['x_hat'].squeeze().cpu().detach().numpy()
    comp_img = np.transpose(comp_img, (1, 2, 0))
    comp_img = rgb2ycbcr(comp_img)
    return comp_img, bits


def compress_cheng2020_anchor(qual_level, img):
    img = ycbcr2rgb(img) #data range should be between [0, 1]
    #img = np.transpose(img, (2, 0, 1))
    # convert to PIL image
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)
    #breakpoint()
    model = cheng2020_anchor(quality=qual_level+1, metric='ms-ssim', pretrained=True).eval().to(device)
    model.eval()
    model.update(force=True)
    #img = torch.tensor(img).unsqueeze(0).type(torch.float64)
    #shape_img = img.shape
    comp_img = model(x)
    #comp_img = model.decompress(comp_img, shape_img).squeeze().numpy()
    bits = compute_bpp(comp_img)
    comp_img = comp_img['x_hat'].squeeze().cpu().detach().numpy()
    comp_img = np.transpose(comp_img, (1, 2, 0))
    comp_img = rgb2ycbcr(comp_img)
    return comp_img, bits

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
        comp_img_bmshj2018_factorized, bits_bmshj2018_factorized_tmp = compress_bmshj2018_factorized(j, img)
        comp_img_cheng2020_anchor, bits_cheng2020_anchor_tmp = compress_cheng2020_anchor(j, img)


        bits_img.append(bits_tmp)
        bits_img_nsqjpeg.append(bits_nsqjpeg_tmp)
        bits_img_bmshj2018_factorized.append(bits_bmshj2018_factorized_tmp)
        bits_img_cheng2020_anchor.append(bits_cheng2020_anchor_tmp)

        mse_vals[j, i], lpips_vals[j, i], ssim_vals[j, i], msssim_vals[j, i], bris_vals[j, i] = evaluate_metrics(img, comp_img_jpeg)
        mse_vals_nsqjpeg[j, i], lpips_vals_nsqjpeg[j, i], ssim_vals_nsqjpeg[j, i], msssim_vals_nsqjpeg[j, i], bris_vals_nsqjpeg[j, i] = evaluate_metrics(img, comp_img_nsqjpeg)
        mse_vals_bmshj2018_factorized[j, i], lpips_vals_bmshj2018_factorized[j, i], ssim_vals_bmshj2018_factorized[j, i], msssim_vals_bmshj2018_factorized[j, i], bris_vals_bmshj2018_factorized[j, i] = evaluate_metrics(img, comp_img_bmshj2018_factorized)
        mse_vals_cheng2020_anchor[j, i], lpips_vals_cheng2020_anchor[j, i], ssim_vals_cheng2020_anchor[j, i], msssim_vals_cheng2020_anchor[j, i], bris_vals_cheng2020_anchor[j, i] = evaluate_metrics(img, comp_img_cheng2020_anchor)

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
    bits_bmshj2018_factorized.append(bits_img_bmshj2018_factorized)
    bits_chen2020_anchor.append(bits_img_cheng2020_anchor)

    total_bits = np.array([bits_img[j] for j in range(nqs)])/(img.shape[0]*img.shape[1])
    total_bits_nsqjpeg = np.array([bits_img_nsqjpeg[j] for j in range(nqs)])/(img.shape[0]*img.shape[1])
    total_bits_bmshj2018_factorized = np.array([bits_img_bmshj2018_factorized[j] for j in range(nqs)])
    total_bits_cheng2020_anchor = np.array([bits_img_cheng2020_anchor[j] for j in range(nqs)])

    Qmtx = nsqjpeg.Qmtx
    plt.figure(figsize=(20, 10))
    plt.subplot(2, 2, 1)
    plt.plot(total_bits, mse_vals[:, i], label='JPEG',linewidth=3)
    plt.plot(total_bits_nsqjpeg, mse_vals_nsqjpeg[:, i], label='NSQJPEG', linewidth=3)
    plt.plot(total_bits_bmshj2018_factorized, mse_vals_bmshj2018_factorized[:, i], label='bmshj2018_factorized', linewidth=3)
    plt.plot(total_bits_cheng2020_anchor, mse_vals_cheng2020_anchor[:, i], label='cheng2020_anchor', linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 16)
    plt.title('MSE', fontsize=16)
    plt.subplot(2, 2, 2)
    plt.plot(total_bits, msssim_vals[:, i], label='JPEG', linewidth=3)
    plt.plot(total_bits_nsqjpeg, msssim_vals_nsqjpeg[:, i], label='NSQJPEG', linewidth=3)
    plt.plot(total_bits_bmshj2018_factorized, msssim_vals_bmshj2018_factorized[:, i], label='bmshj2018_factorized', linewidth=3)
    plt.plot(total_bits_cheng2020_anchor, msssim_vals_cheng2020_anchor[:, i], label='cheng2020_anchor', linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 16)
    plt.title('MSSSIM', fontsize=16)
    plt.subplot(2, 2, 3)
    plt.plot(total_bits, ssim_vals[:, i], label='JPEG', linewidth=3)
    plt.plot(total_bits_nsqjpeg, ssim_vals_nsqjpeg[:, i], label='NSQJPEG', linewidth=3)
    plt.plot(total_bits_bmshj2018_factorized, ssim_vals_bmshj2018_factorized[:, i], label='bmshj2018_factorized', linewidth=3)
    plt.plot(total_bits_cheng2020_anchor, ssim_vals_cheng2020_anchor[:, i], label='cheng2020_anchor', linewidth=3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize = 16)
    plt.title('SSIM', fontsize=16)
    plt.subplot(2, 2, 4)
    plt.plot(total_bits, lpips_vals[:, i], label='JPEG', linewidth=3)
    plt.plot(total_bits_nsqjpeg, lpips_vals_nsqjpeg[:, i], label='NSQJPEG', linewidth=3)
    plt.plot(total_bits_bmshj2018_factorized, lpips_vals_bmshj2018_factorized[:, i], label='bmshj2018_factorized', linewidth=3)
    plt.plot(total_bits_cheng2020_anchor, lpips_vals_cheng2020_anchor[:, i], label='cheng2020_anchor', linewidth=3)
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
    touple_mse_bmshj2018_factorized = get_touples(mse_vals_bmshj2018_factorized[:, i], total_bits_bmshj2018_factorized)
    touple_mse_cheng2020_anchor = get_touples(mse_vals_cheng2020_anchor[:, i], total_bits_cheng2020_anchor)

    touple_lpips_jpeg = get_touples(lpips_vals[:, i], total_bits)
    touple_lpips_nsqjpeg = get_touples(lpips_vals_nsqjpeg[:, i], total_bits_nsqjpeg)
    touple_lpips_bmshj2018_factorized = get_touples(lpips_vals_bmshj2018_factorized[:, i], total_bits_bmshj2018_factorized)
    touple_lpips_cheng2020_anchor = get_touples(lpips_vals_cheng2020_anchor[:, i], total_bits_cheng2020_anchor)

    touple_ssim_jpeg = get_touples(ssim_vals[:, i], total_bits)
    touple_ssim_nsqjpeg = get_touples(ssim_vals_nsqjpeg[:, i], total_bits_nsqjpeg)
    touple_ssim_bmshj2018_factorized = get_touples(ssim_vals_bmshj2018_factorized[:, i], total_bits_bmshj2018_factorized)
    touple_ssim_cheng2020_anchor = get_touples(ssim_vals_cheng2020_anchor[:, i], total_bits_cheng2020_anchor)

    touple_msssim_jpeg = get_touples(msssim_vals[:, i], total_bits)
    touple_msssim_nsqjpeg = get_touples(msssim_vals_nsqjpeg[:, i], total_bits_nsqjpeg)
    touple_msssim_bmshj2018_factorized = get_touples(msssim_vals_bmshj2018_factorized[:, i], total_bits_bmshj2018_factorized)
    touple_msssim_cheng2020_anchor = get_touples(msssim_vals_cheng2020_anchor[:, i], total_bits_cheng2020_anchor)

    touple_bris_jpeg = get_touples(bris_vals[:, i], total_bits)
    touple_bris_nsqjpeg = get_touples(bris_vals_nsqjpeg[:, i], total_bits_nsqjpeg)
    touple_bris_bmshj2018_factorized = get_touples(bris_vals_bmshj2018_factorized[:, i], total_bits_bmshj2018_factorized)
    touple_bris_cheng2020_anchor = get_touples(bris_vals_cheng2020_anchor[:, i], total_bits_cheng2020_anchor)


    bdrate_mse_nsqjpeg[i] = bdrate(touple_mse_jpeg, touple_mse_nsqjpeg)
    bdrate_lpips_nsqjpeg[i] = bdrate(touple_lpips_jpeg, touple_lpips_nsqjpeg)
    bdrate_ssim_nsqjpeg[i] = bdrate(touple_ssim_jpeg, touple_ssim_nsqjpeg)
    bdrate_msssim_nsqjpeg[i] = bdrate(touple_msssim_jpeg, touple_msssim_nsqjpeg)
    bdrate_bris_nsqjpeg[i] = bdrate(touple_bris_jpeg, touple_bris_nsqjpeg)

    bdrate_mse_bmshj2018_factorized[i] = bdrate(touple_mse_jpeg, touple_mse_bmshj2018_factorized)
    bdrate_lpips_bmshj2018_factorized[i] = bdrate(touple_lpips_jpeg, touple_lpips_bmshj2018_factorized)
    bdrate_ssim_bmshj2018_factorized[i] = bdrate(touple_ssim_jpeg, touple_ssim_bmshj2018_factorized)
    bdrate_msssim_bmshj2018_factorized[i] = bdrate(touple_msssim_jpeg, touple_msssim_bmshj2018_factorized)
    bdrate_bris_bmshj2018_factorized[i] = bdrate(touple_bris_jpeg, touple_bris_bmshj2018_factorized)
    
    bdrate_mse_cheng2020_anchor[i] = bdrate(touple_mse_jpeg, touple_mse_cheng2020_anchor)
    bdrate_lpips_cheng2020_anchor[i] = bdrate(touple_lpips_jpeg, touple_lpips_cheng2020_anchor)
    bdrate_ssim_cheng2020_anchor[i] = bdrate(touple_ssim_jpeg, touple_ssim_cheng2020_anchor)
    bdrate_msssim_cheng2020_anchor[i] = bdrate(touple_msssim_jpeg, touple_msssim_cheng2020_anchor)
    bdrate_bris_cheng2020_anchor[i] = bdrate(touple_bris_jpeg, touple_bris_cheng2020_anchor)
    
    nsqjpeg_mse = get_mean_format(bdrate_mse_nsqjpeg[i])
    nsqjpeg_lpips = get_mean_format(bdrate_lpips_nsqjpeg[i])
    nsqjpeg_ssim = get_mean_format(bdrate_ssim_nsqjpeg[i])
    nsqjpeg_msssim = get_mean_format(bdrate_msssim_nsqjpeg[i])
    nsqjpeg_bris = get_mean_format(bdrate_bris_nsqjpeg[i])

    bmshj2018_factorized_mse = get_mean_format(bdrate_mse_bmshj2018_factorized[i])
    bmshj2018_factorized_lpips = get_mean_format(bdrate_lpips_bmshj2018_factorized[i])
    bmshj2018_factorized_ssim = get_mean_format(bdrate_ssim_bmshj2018_factorized[i])
    bmshj2018_factorized_msssim = get_mean_format(bdrate_msssim_bmshj2018_factorized[i])
    bmshj2018_factorized_bris = get_mean_format(bdrate_bris_bmshj2018_factorized[i])

    cheng2020_anchor_mse = get_mean_format(bdrate_mse_cheng2020_anchor[i])
    cheng2020_anchor_lpips = get_mean_format(bdrate_lpips_cheng2020_anchor[i])
    cheng2020_anchor_ssim = get_mean_format(bdrate_ssim_cheng2020_anchor[i])
    cheng2020_anchor_msssim = get_mean_format(bdrate_msssim_cheng2020_anchor[i])
    cheng2020_anchor_bris = get_mean_format(bdrate_bris_cheng2020_anchor[i])

    print('nsqjpeg image ' + str(i+1) + ' MSE: ' + nsqjpeg_mse + ' SSIM: ' + nsqjpeg_ssim + ' LPIPS: ' + nsqjpeg_lpips + ' MSSSIM: ' + nsqjpeg_msssim + ' BRIS: ' + nsqjpeg_bris)
    print('bmshj2018_factorized image ' + str(i+1) + ' MSE: ' + bmshj2018_factorized_mse + ' SSIM: ' + bmshj2018_factorized_ssim + ' LPIPS: ' + bmshj2018_factorized_lpips + ' MSSSIM: ' + bmshj2018_factorized_msssim + ' BRIS: ' + bmshj2018_factorized_bris)
    print('cheng2020_anchor image ' + str(i+1) + ' MSE: ' + cheng2020_anchor_mse + ' SSIM: ' + cheng2020_anchor_ssim + ' LPIPS: ' + cheng2020_anchor_lpips + ' MSSSIM: ' + cheng2020_anchor_msssim + ' BRIS: ' + cheng2020_anchor_bris)

    nsqjpeg_mse = get_mean_format(bdrate_mse_nsqjpeg[:i+1])
    nsqjpeg_lpips = get_mean_format(bdrate_lpips_nsqjpeg[:i+1])
    nsqjpeg_ssim = get_mean_format(bdrate_ssim_nsqjpeg[:i+1])
    nsqjpeg_msssim = get_mean_format(bdrate_msssim_nsqjpeg[:i+1])
    nsqjpeg_bris = get_mean_format(bdrate_bris_nsqjpeg[:i+1])

    bmshj2018_factorized_mse = get_mean_format(bdrate_mse_bmshj2018_factorized[:i+1])
    bmshj2018_factorized_lpips = get_mean_format(bdrate_lpips_bmshj2018_factorized[:i+1])
    bmshj2018_factorized_ssim = get_mean_format(bdrate_ssim_bmshj2018_factorized[:i+1])
    bmshj2018_factorized_msssim = get_mean_format(bdrate_msssim_bmshj2018_factorized[:i+1])
    bmshj2018_factorized_bris = get_mean_format(bdrate_bris_bmshj2018_factorized[:i+1])

    cheng2020_anchor_mse = get_mean_format(bdrate_mse_cheng2020_anchor[:i+1])
    cheng2020_anchor_lpips = get_mean_format(bdrate_lpips_cheng2020_anchor[:i+1])
    cheng2020_anchor_ssim = get_mean_format(bdrate_ssim_cheng2020_anchor[:i+1])
    cheng2020_anchor_msssim = get_mean_format(bdrate_msssim_cheng2020_anchor[:i+1])
    cheng2020_anchor_bris = get_mean_format(bdrate_bris_cheng2020_anchor[:i+1])

    print('Mean nsqjpeg MSE: ' + nsqjpeg_mse + ' SSIM: ' + nsqjpeg_ssim + ' LPIPS: ' + nsqjpeg_lpips + ' MSSSIM: ' + nsqjpeg_msssim + ' BRIS: ' + nsqjpeg_bris)
    print('Mean bmshj2018_factorized MSE: ' + bmshj2018_factorized_mse + ' SSIM: ' + bmshj2018_factorized_ssim + ' LPIPS: ' + bmshj2018_factorized_lpips + ' MSSSIM: ' + bmshj2018_factorized_msssim + ' BRIS: ' + bmshj2018_factorized_bris)
    print('Mean cheng2020_anchor MSE: ' + cheng2020_anchor_mse + ' SSIM: ' + cheng2020_anchor_ssim + ' LPIPS: ' + cheng2020_anchor_lpips + ' MSSSIM: ' + cheng2020_anchor_msssim + ' BRIS: ' + cheng2020_anchor_bris)