from utils.utils_lpit import bdrate
from my_codecs.JPEG_class import JPEG
from my_codecs.NSQJPEG import NSQJPEG

import numpy as np
import os
import matplotlib.pyplot as plt
from compute_q.compute_Q_machines import compute_Q_machines as compute_Q_class

import torch
from torchvision import datasets, transforms
from utils.q_utils import resnet_loss
from utils.utils_lpit import rgb2ycbcr, ycbcr2rgb

classifier = torch.load('sketch/models/transfer_insects.pth')
classifier.eval()

train_name = 'train'
test_name = 'val'

n_cwd = 4
true_N = (224, 224)
nqs = 12
N = 8

flag_uniform = True

jpeg = JPEG(nqs, uniform=flag_uniform, N=8)
compute_Q_obj = compute_Q_class(true_N=true_N, n_cwd=n_cwd, N=8)
nsqjpeg = NSQJPEG(compute_Q_obj, nqs=nqs, N=N, uniform=True)

depth = 3

def compress_JPEG(qual_lev, img):
    trans, bits = jpeg.compress(img, depth, qual_lev)
    comp_img = jpeg.uncompress(trans, depth, qual_lev)
    return comp_img, bits


def compress_NSQJPEG(qual_lev, img):
    if (depth == 1):
        trans, bits = nsqjpeg.compress(img, depth, qual_lev)
        comp_img = nsqjpeg.uncompress(trans, depth, qual_lev)
    else:
        comp_img = np.zeros_like(img)
        bits = 0

        # compress channel-wise. And only count the overhead for Q once--we use the same Q.

        y_img = img[:, :, 0].squeeze()
        trans, bits_1 = nsqjpeg.compress(y_img, 1, qual_lev)
        bits += bits_1.bits_over
        comp_img[:, :, 0] = nsqjpeg.uncompress(trans, 1, qual_lev)

        cb_img = img[:, :, 1].squeeze()
        trans, bits_1 = nsqjpeg.compress(cb_img, 1, qual_lev)
        bits += bits_1.bits_over - nsqjpeg.overhead_bits
        comp_img[:, :, 1] = nsqjpeg.uncompress(trans, 1, qual_lev)

        cr_img = img[:, :, 2].squeeze()
        trans, bits_1 = nsqjpeg.compress(cr_img, 1, qual_lev)
        bits += bits_1.bits_over - nsqjpeg.overhead_bits
        comp_img[:, :, 2] = nsqjpeg.uncompress(trans, 1, qual_lev)
    return comp_img, bits

data_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def evaluate_classifier(img1, label):
    img1 = img1.transpose((2, 0, 1))
    img1 = torch.from_numpy(img1).unsqueeze(0).float()
    img1 = data_normalize(img1)    
    with torch.no_grad():
        classifier.eval()
        outputs = classifier(img1)
    _, preds = torch.max(outputs, 1)
    return preds.item() == label.item()


def evaluate_metrics(img1, img2):
    psnr = 10*np.log10(255**2/np.mean(np.square(255*(img1 - img2))))
    img1 = img1.transpose((2, 0, 1))
    img2 = img2.transpose((2, 0, 1))
    img1 = torch.from_numpy(img1).unsqueeze(0).float()
    img2 = torch.from_numpy(img2).unsqueeze(0).float()
    img1 = data_normalize(img1)
    img2 = data_normalize(img2)
    loss_res = resnet_loss(img1, img2)
    psnr_emb = 10*np.log10(1/loss_res.item())
    return psnr, psnr_emb


def get_touples(mag, bits):
    arr_out = np.zeros((len(mag), 2))
    order = np.argsort(bits)
    arr_out[:, 0] = bits[order]
    arr_out[:, 1] = mag[order]
    return arr_out


def get_mean_format(data):
    mean = np.round(np.mean(data), 2)
    return '{}'.format(mean)


num_images = 30

data_transforms_nonorm = {
    train_name: transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]),
    test_name: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ]),
}


data_dir = 'Images/hymenoptera_data'

#set random seed
torch.manual_seed(3)

image_datasets_nonorm = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms_nonorm[x])
                  for x in [train_name, test_name]}

dataloaders_nonorm = {x: torch.utils.data.DataLoader(image_datasets_nonorm[x], batch_size=1,
                                                shuffle=True, num_workers=4)
                for x in [train_name, test_name]}

dataset_sizes = {x: len(image_datasets_nonorm[x]) for x in [train_name, test_name]}

bdrate_emb_nsqjpeg = np.zeros(num_images)
bdrate_mse_nsqjpeg = np.zeros_like(bdrate_emb_nsqjpeg)

right_vals = np.zeros((nqs, num_images))
right_vals_jpeg = np.zeros((nqs, num_images))
right_vals_nsqjpeg = np.zeros_like(right_vals)


mse_vals_jpeg = np.zeros((nqs, num_images))
mse_vals_nsqjpeg = np.zeros_like(mse_vals_jpeg)

emb_vals_jpeg = np.zeros_like(mse_vals_jpeg)
emb_vals_nsqjpeg = np.zeros_like(mse_vals_jpeg)

# create array to store bits
bits = np.zeros((nqs, num_images))
bits_nsqjpeg = np.zeros_like(bits)

for i in range(num_images):

    img, label = next(iter(dataloaders_nonorm[test_name]))

    img_norm = data_normalize(img)

    img = img[0, :, :, :].numpy()
    img = img.transpose((1, 2, 0))

    img_norm = img_norm[0, :, :, :].numpy()

    nsqjpeg.set_Q(img_norm, one_depth=False)
    jpeg.set_Q(img)

    right_vals[:, i] = evaluate_classifier(img, label)

    print('Image: ', (i), 'Number of images: ', num_images, 'Label: ', label.item(), 'Original prediction right: ', right_vals[0, i])
    img_yuv = rgb2ycbcr(img)
    
    inds = nsqjpeg.ind_closest

    img_block = np.zeros(true_N)
    
    for p in range(0, true_N[0]//N):
        for q in range(0, true_N[1]//N):
            indi = inds[p, q].astype(int)
            img_block[p*N:(p+1)*N, q*N:(q+1)*N] = nsqjpeg.centroids[indi, :, :].squeeze()

    plt.figure()
    plt.subplot(1, 3, 1)
    img_show = ycbcr2rgb(img_yuv)
    im = plt.imshow(img_show/255)
    plt.colorbar(im)
    plt.title('Original')
    # remove ticks
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 2)
    im = plt.imshow(nsqjpeg.Qmtx)
    plt.colorbar(im)
    plt.title('Qmtx')
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 3, 3)
    im = plt.imshow(img_block)
    plt.colorbar(im)
    plt.title('NSQJPEG block')
    plt.xticks([])
    plt.yticks([])
    plt.show()
