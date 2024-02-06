from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect, ycbcr2rgb, rgb2ycbcr
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_msssim as compute_Q_class
from handling_q.q_ops import q_ops_msssim as compute_ops_class
from handling_q.q_ops_noco import q_ops_ssim as compute_ops_class_noco
import random
from utils.q_utils_jax import compute_LPIPS_color
from iqa_funs.jax_ssim import jax_ssim_eval as ssim_mod
from iqa_funs.jax_ssim import jax_msssim as ms_ssim_mod
import jax.numpy as jnp
from PIL import Image
from torchvision import transforms
import torch
from compressai.zoo import bmshj2018_factorized

ssim_func = lambda x, y: -10*np.log10(ssim_mod(x, y))
ms_ssim_func = lambda x, y: -10*np.log10(ms_ssim_mod(x, y))

# set the scaling factor in NSQJPEG to 50 to obtain these curves

device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_cwd = 4
true_N = (1024, 1024)
nqs = 12
N = 8

flag_uniform = True

jpeg = JPEG(nqs, uniform=flag_uniform, N=8)
compute_Q_obj = compute_Q_class(true_N, sampling_depth=16)
q_ops_obj = compute_ops_class(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
q_ops_obj_noco = compute_ops_class_noco(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
nsqjpeg = NSQJPEG(compute_Q_obj, q_ops_obj, nqs=nqs, N=N, uniform=flag_uniform, quant_scal=50)


def compress_JPEG(qual_lev, img):
    trans, bits, bits_arr = jpeg.compress_420(img, qual_lev)
    comp_img = jpeg.uncompress_420(trans, qual_lev)
    return comp_img, bits, bits_arr


def compress_NSQJPEG(qual_lev, img):
    trans, bits, bits_arr = nsqjpeg.compress_420(img, qual_lev)
    comp_img = nsqjpeg.uncompress_420(trans, qual_lev)
    return comp_img, bits, bits_arr


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    likelihoods = out_net['likelihoods']['y']
    output = (torch.log(likelihoods).sum() / (-np.log(2) * num_pixels)).item()
    return output


def compress_bmshj2018_factorized(qual_level, img):
    img = ycbcr2rgb(img)
    # convert to PIL image
    img = Image.fromarray(img.astype('uint8'), 'RGB') # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    x = transforms.ToTensor()(img).unsqueeze(0).to(device)    
    #breakpoint()
    model = bmshj2018_factorized(quality=qual_level+1, metric='ms-ssim', pretrained=True).eval().to(device)
    model.eval()
    model.update(force=True) # is this step really needed?
    comp_img = model(x)
    bits = compute_bpp(comp_img)
    comp_img = comp_img['x_hat'].squeeze().cpu().detach().numpy()
    comp_img = np.transpose(comp_img, (1, 2, 0))
    comp_img = rgb2ycbcr(comp_img)
    return comp_img, bits


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
        msssim_score[i] = ms_ssim_func(ten_img1, ten_img2)
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
    mean = np.round(np.mean(data), 3)
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
mse_vals_bmshj2018_factorized = np.zeros_like(mse_vals)

lpips_vals = np.zeros_like(mse_vals)
lpips_vals_nsqjpeg = np.zeros_like(mse_vals)
lpips_vals_bmshj2018_factorized = np.zeros_like(mse_vals)

ssim_vals = np.zeros_like(mse_vals)
ssim_vals_nsqjpeg = np.zeros_like(mse_vals)
ssim_vals_bmshj2018_factorized = np.zeros_like(mse_vals)

msssim_vals = np.zeros_like(mse_vals)
msssim_vals_nsqjpeg = np.zeros_like(mse_vals)
msssim_vals_bmshj2018_factorized = np.zeros_like(mse_vals)

# create array to store bits
bits = []
bits_nsqjpeg = []

for i in range(num_images):
    
    
    if (i != 19):
        continue

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
    bits_img_bmshj2018_factorized = []

    mse_balle = []
    lpips_balle = []
    ssim_balle = []
    msssim_balle = []

    
    for j in range(8):
        comp_img_bmshj2018_factorized, bits_bmshj2018_factorized_tmp = compress_bmshj2018_factorized(j, img)
        bits_img_bmshj2018_factorized.append(bits_bmshj2018_factorized_tmp)
        mse_val, lpips_val, ssim_val, msssim_val = evaluate_metrics(img, comp_img_bmshj2018_factorized)
        mse_balle.append(mse_val)
        lpips_balle.append(lpips_val)
        ssim_balle.append(ssim_val)
        msssim_balle.append(msssim_val)
        img_rgb = ycbcr2rgb(comp_img_bmshj2018_factorized)
        im = Image.fromarray(img_rgb)
        im.save("week_4/data/compressed/balle_"+str(j)+".png")


    total_bits_bmshj2018_factorized = np.array(bits_img_bmshj2018_factorized)
    mse_vals_bmshj2018_factorized = np.array(mse_balle)
    lpips_vals_bmshj2018_factorized = np.array(lpips_balle)
    ssim_vals_bmshj2018_factorized = np.array(ssim_balle)
    msssim_vals_bmshj2018_factorized = np.array(msssim_balle)

    for j in range(nqs):

        qual_idx = j
        comp_img_jpeg, bits_tmp, bits_arr_1 = compress_JPEG(j, img)
        comp_img_nsqjpeg, bits_nsqjpeg_tmp, bits_arr_2 = compress_NSQJPEG(j, img)

        bits_img.append(bits_tmp)
        bits_img_nsqjpeg.append(bits_nsqjpeg_tmp)

        mse_vals[j, i], lpips_vals[j, i], ssim_vals[j, i], msssim_vals[j, i] = evaluate_metrics(img, comp_img_jpeg)
        mse_vals_nsqjpeg[j, i], lpips_vals_nsqjpeg[j, i], ssim_vals_nsqjpeg[j, i], msssim_vals_nsqjpeg[j, i] = evaluate_metrics(img, comp_img_nsqjpeg)

        img_rgb = ycbcr2rgb(comp_img_jpeg)
        im = Image.fromarray(img_rgb)
        im.save("week_4/data/compressed/jpeg_"+str(j)+".png")
        img_rgb = ycbcr2rgb(comp_img_nsqjpeg)
        im = Image.fromarray(img_rgb)
        im.save("week_4/data/compressed/nsqjpeg_"+str(j)+".png")

    bits.append(bits_img)
    bits_nsqjpeg.append(bits_img_nsqjpeg)

    total_bits = np.array([bits_img[j] for j in range(nqs)])/(img.shape[0]*img.shape[1])
    total_bits_nsqjpeg = np.array([bits_img_nsqjpeg[j] for j in range(nqs)])/(img.shape[0]*img.shape[1])
    
    np.savez('week_4/data/pretty_rd.npz', total_bits=total_bits, total_bits_nsqjpeg=total_bits_nsqjpeg, total_bits_bmshj2018_factorized=total_bits_bmshj2018_factorized, mse_vals=mse_vals, mse_vals_nsqjpeg=mse_vals_nsqjpeg, mse_vals_bmshj2018_factorized=mse_vals_bmshj2018_factorized, lpips_vals=lpips_vals, lpips_vals_nsqjpeg=lpips_vals_nsqjpeg, lpips_vals_bmshj2018_factorized=lpips_vals_bmshj2018_factorized, ssim_vals=ssim_vals, ssim_vals_nsqjpeg=ssim_vals_nsqjpeg, ssim_vals_bmshj2018_factorized=ssim_vals_bmshj2018_factorized, msssim_vals=msssim_vals, msssim_vals_nsqjpeg=msssim_vals_nsqjpeg, msssim_vals_bmshj2018_factorized=msssim_vals_bmshj2018_factorized)

    arr = np.load('week_4/data/pretty_rd.npz')

    total_bits = arr['total_bits']
    total_bits_nsqjpeg = arr['total_bits_nsqjpeg']
    total_bits_bmshj2018_factorized = arr['total_bits_bmshj2018_factorized']
    mse_vals = arr['mse_vals']
    mse_vals_nsqjpeg = arr['mse_vals_nsqjpeg']
    mse_vals_bmshj2018_factorized = arr['mse_vals_bmshj2018_factorized']
    lpips_vals = arr['lpips_vals']
    lpips_vals_nsqjpeg = arr['lpips_vals_nsqjpeg']
    lpips_vals_bmshj2018_factorized = arr['lpips_vals_bmshj2018_factorized']
    ssim_vals = arr['ssim_vals']
    ssim_vals_nsqjpeg = arr['ssim_vals_nsqjpeg']
    ssim_vals_bmshj2018_factorized = arr['ssim_vals_bmshj2018_factorized']
    msssim_vals = arr['msssim_vals']
    msssim_vals_nsqjpeg = arr['msssim_vals_nsqjpeg']
    msssim_vals_bmshj2018_factorized = arr['msssim_vals_bmshj2018_factorized']
    
    mse_vals_nsqjpeg = mse_vals_nsqjpeg[:, i]
    mse_vals = mse_vals[:, i]

    lpips_vals_nsqjpeg = lpips_vals_nsqjpeg[:, i]
    lpips_vals = lpips_vals[:, i]
    
    ssim_vals_nsqjpeg = ssim_vals_nsqjpeg[:, i]
    ssim_vals = ssim_vals[:, i]
    
    msssim_vals_nsqjpeg = msssim_vals_nsqjpeg[:, i]
    msssim_vals = msssim_vals[:, i]

    plt.figure(figsize=(20, 5))
    for j in range(8):
        plt.subplot(1, 8, j+1)
        plt.imshow(np.array(Image.open('week_4/data/compressed/balle_'+str(j)+'.png')))
        plt.axis('off')
        plt.title('BPP: '+get_mean_format(total_bits_bmshj2018_factorized[j]) + 'MS-SSIM: '+get_mean_format(msssim_vals_bmshj2018_factorized[j]))
    plt.suptitle('Balle')

    plt.figure(figsize=(20, 5))
    for j in range(nqs):
        plt.subplot(1, nqs, j+1)
        plt.imshow(np.array(Image.open('week_4/data/compressed/jpeg_'+str(j)+'.png')))
        plt.axis('off')
        plt.title('BPP: '+get_mean_format(total_bits[j]) + 'MS-SSIM: '+get_mean_format(msssim_vals[j]))
    plt.suptitle('JPEG')

    plt.figure(figsize=(20, 5))
    for j in range(nqs):
        plt.subplot(1, nqs, j+1)
        plt.imshow(np.array(Image.open('week_4/data/compressed/nsqjpeg_'+str(j)+'.png')))
        plt.axis('off')
        plt.title('BPP: '+get_mean_format(total_bits_nsqjpeg[j]) + '. MS-SSIM: '+get_mean_format(msssim_vals_nsqjpeg[j]))
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    j = 10
    img = np.array(Image.open('week_4/data/compressed/jpeg_'+str(j)+'.png'))
    plt.imshow(img)
    plt.axis('off')
    msssim_v = 1-10**(-msssim_vals[j]/10)
    plt.title('JPEG. BPP: '+get_mean_format(total_bits[j]) + '. MS-SSIM: '+get_mean_format(msssim_v))
    plt.subplot(1, 3, 2)
    j = 9
    img = np.array(Image.open('week_4/data/compressed/nsqjpeg_'+str(j)+'.png'))
    plt.imshow(img)
    plt.axis('off')
    msssim_v = 1-10**(-msssim_vals_nsqjpeg[j]/10)
    plt.title('MS-SSIM JPEG. BPP: '+get_mean_format(total_bits_nsqjpeg[j]) + '. MS-SSIM: '+get_mean_format(msssim_v))
    plt.subplot(1, 3, 3)
    j = 6
    img = np.array(Image.open('week_4/data/compressed/balle_'+str(j)+'.png'))
    plt.imshow(img)
    plt.axis('off')
    msssim_v = 1-10**(-msssim_vals_bmshj2018_factorized[j]/10)
    plt.title('Ballé et al. 2018. BPP: '+get_mean_format(total_bits_bmshj2018_factorized[j]) + '. MS-SSIM: '+get_mean_format(msssim_v))
    plt.show()