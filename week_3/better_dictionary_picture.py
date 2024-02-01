from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG
from utils.utils_lpit import read_image_resize_rect, ycbcr2rgb
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_jax import compute_Q_ssim as compute_Q_class
from handling_q.q_ops import q_ops_ssim as compute_ops_class

import random
import torch
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
flag_uniform = True

compute_Q_obj = compute_Q_class(true_N, sampling_depth=32)
q_ops_obj = compute_ops_class(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)
nsqjpeg = NSQJPEG(compute_Q_obj, q_ops_obj, nqs=nqs, N=N, uniform=flag_uniform)
jpeg = JPEG(nqs, uniform=flag_uniform, N=8)

path = 'Images/CLIC/Testing/'
dirs = os.listdir(path)
num_images = 20
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

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

    img_show = ycbcr2rgb(img)

    Qmtx = nsqjpeg.Qmtx
    Q_2 = nsqjpeg.q_ops_obj.Q_list[0]
    Q_4 = nsqjpeg.q_ops_obj.Q_list[2]
    Q_8 = nsqjpeg.q_ops_obj.Q_list[-1]

    q_2_br = nsqjpeg.q_ops_obj.rate_list[0]
    q_4_br = nsqjpeg.q_ops_obj.rate_list[2]
    q_8_br = nsqjpeg.q_ops_obj.rate_list[-1]

    plt.figure(figsize=(25, 3))
    plt.subplot(1, 5, 1)
    plt.imshow(img_show/255)
    plt.title('Original', fontsize=16)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 5, 2)
    im = plt.imshow(Qmtx, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('IDQD matrix', fontsize=16)
    plt.colorbar(im)
    plt.subplot(1, 5, 3)
    im = plt.imshow(Q_2, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('CW: 2 . OH: ' + str(int(q_2_br)) + ' bits', fontsize=16)
    plt.colorbar(im)
    plt.subplot(1, 5, 4)
    im = plt.imshow(Q_4, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('CW: 4. OH: ' + str(int(q_4_br)) + ' bits', fontsize=16)
    plt.colorbar(im)
    plt.subplot(1, 5, 5)
    im = plt.imshow(Q_8, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.title('CW: 8. OH: ' + str(int(q_8_br)) + ' bits', fontsize=16)
    plt.colorbar(im)

    # set them to tight
    plt.tight_layout()
    plt.show()

