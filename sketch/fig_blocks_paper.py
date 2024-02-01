from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
from utils.utils_lpit import read_image_resize_rect
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_pytorch import compute_Q_ssim as compute_Q_class
from utils.utils_lpit import ycbcr2rgb

import random

n_cwd = 4
true_N = (512, 512)
nqs = 6
N = 8

###### REMEMBER THAT NSQJPEG IS MODIFIED TO NOT INCLUDE RLGR ENCODING ######
###### 
flag_uniform = True

jpeg = JPEG(nqs, uniform=flag_uniform, N=8)
compute_Q_obj = compute_Q_class(true_N=true_N, n_cwd=n_cwd, N=8)
nsqjpeg = NSQJPEG(compute_Q_obj, nqs=nqs, N=N, uniform=flag_uniform)



def get_mean_format(data):
    mean = np.round(np.mean(data), 2)
    return '{}'.format(mean)


path = 'Images/CLIC/Testing/'
dirs = os.listdir(path)
num_images = 100
random.seed(0)
random.shuffle(dirs)
dirs = dirs[:num_images]

for i in range(num_images):
    if not dirs[i].__contains__('nomao-saeki-33553'):
        print(dirs[i])
        continue

    fullpath = os.path.join(path,dirs[i])  
    img, depth = read_image_resize_rect(fullpath, true_N)
    # replace UGC by PBD
    img_3d = ycbcr2rgb(img)
    img = img[:, :, 0]
    depth = 1
    nsqjpeg.set_Q(img)

    Qmtx = nsqjpeg.Qmtx

    indis = nsqjpeg.ind_closest
    centroids = compute_Q_obj.get_centroids()

    Q_blocks = np.zeros_like(img)

    for p in range(indis.shape[0]):
        for q in range(indis.shape[1]):
            ind = int(indis[p, q])
            Q_blocks[p*N:(p+1)*N, q*N:(q+1)*N] = centroids[ind, :, :]

    plt.figure()

    im = plt.imshow(img_3d)
    plt.xticks([])
    plt.yticks([])
    plt.title('Image', fontsize=16)


    plt.figure()

    plt.subplot(2, 2, 1)
    im = plt.imshow(centroids[0], cmap='gray')
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 2)
    im = plt.imshow(centroids[1], cmap='gray')
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(2, 2, 3)
    im = plt.imshow(centroids[2], cmap='gray')
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(2, 2, 4)
    im = plt.imshow(centroids[3], cmap='gray')
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])

    plt.suptitle('Codewords', fontsize=16)


    plt.figure()

    im = plt.imshow(Qmtx, cmap='gray')
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    plt.title('Weights', fontsize=16)

    plt.subplot(1, 2, 2)
    im = plt.imshow(Q_blocks, cmap='gray')
    plt.colorbar(im)
    plt.xticks([])
    plt.yticks([])
    plt.title('Quantized weights', fontsize=16)

    plt.show()
