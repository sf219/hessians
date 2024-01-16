import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image, ImageOps
from scipy.ndimage.filters import gaussian_filter
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from scipy.stats import vonmises
import math
from scipy.interpolate import interp2d

def quantization_model(Q, q):
    return np.min(Q)*q

def get_quantization_table(N):
    base_Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                [12, 12, 14, 19, 26, 58, 60, 55],
                [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62],
                [18, 22, 37, 56, 68, 109, 103, 77],
                [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101],
                [72, 92, 95, 98, 112, 100, 103, 99]]).astype(np.float64)
    x = np.arange(0, base_Q.shape[0])
    y = np.arange(0, base_Q.shape[1])
    f = interp2d(x, y, base_Q, kind='cubic')
    x_new = np.linspace(0, base_Q.shape[0], N)
    y_new = np.linspace(0, base_Q.shape[1], N)
    base_Q = np.round(f(x_new, y_new))
    base_Q = base_Q / np.min(base_Q)
    return base_Q

def zigzag(n):
    '''zigzag rows'''
    def compare(xy):
        x, y = xy
        return (x + y, -y if (x + y) % 2 else y)
    xs = range(n)
    tmp = {index: n for n, index in enumerate(sorted(
        ((x, y) for x in xs for y in xs),
        key=compare
    ))}
    zz_mtx = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            tp = (j, i)
            zz_mtx[j, i] = tmp[tp]
    zz_mtx = zz_mtx.astype(np.int32)
    return zz_mtx


def ycbcr2rgb(im):
    xform = np.array([[1, 0, 1.402], [1, -0.34414, -.71414], [1, 1.772, 0]])
    rgb = im.astype(np.float32)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

def rgb2ycbcr(rgb):
    # This matrix comes from a formula in Poynton's, "Introduction to
    # Digital Video" (p. 176, equations 9.6).
    # T is from equation 9.6: ycbcr = origT * rgb + origOffset;
    origT = np.array([[65.481, 128.553, 24.966],
                      [-37.797, -74.203, 112],
                      [112, -93.786, -18.214]])
    origOffset = np.array([16, 128, 128])

    # Initialize variables
    isColormap = False

    # Must reshape colormap to be m x n x 3 for transformation
    if rgb.ndim == 2:
        # colormap
        isColormap = True
        colors = rgb.shape[0]
        rgb = rgb.reshape((colors, 1, 3))

    # The formula ycbcr = origT * rgb + origOffset, converts a RGB image in the
    # range [0 1] to a YCbCr image where Y is in the range [16 235], and Cb and
    # Cr are in that range [16 240]. For each class type, we must calculate
    # scaling factors for origT and origOffset so that the input image is
    # scaled between 0 and 1, and so that the output image is in the range of
    # the respective class type.
    if np.issubdtype(rgb.dtype, np.integer):
        if rgb.dtype == np.uint8:
            scaleFactorT = 1/255
            scaleFactorOffset = 1
        elif rgb.dtype == np.uint16:
            scaleFactorT = 257/65535
            scaleFactorOffset = 257
    else:
        scaleFactorT = 1
        scaleFactorOffset = 1

    # The formula ycbcr = origT*rgb + origOffset is rewritten as
    # ycbcr = scaleFactorForT * origT * rgb + scaleFactorForOffset*origOffset.
    # To use np.einsum, we rewrite the formula as ycbcr = T * rgb + offset,
    # where T and offset are defined below.
    T = scaleFactorT * origT
    offset = scaleFactorOffset * origOffset
    ycbcr = np.zeros_like(rgb)
    for p in range(3):
        ycbcr[:, :, p] = T[p, 0]*rgb[:, :, 0] + T[p, 1]*rgb[:, :, 1] + T[p, 2]*rgb[:, :, 2] + offset[p]   

    if isColormap:
        ycbcr = ycbcr.squeeze()
    return ycbcr

def dct_2d(a):
    return dct( dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct_2d(a):
    return idct( idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def read_image_yuv(img_path, im_size=256):
    im = Image.open(img_path)
    im = np.array(im).astype(np.float64)
    im = rgb2ycbcr(im/255)
    im = im[:, :, 0].squeeze()
    depth = 1
    width, height = im.shape
    new_width = im_size
    new_height = im_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    if(depth == 3):
        img = np.array(im)[left:right, top:bottom, :].reshape((new_height, new_width, depth)).squeeze()
    else:
        img = np.array(im)[left:right, top:bottom].reshape((new_height, new_width)).squeeze()
    return img, depth

def read_image(img_path, im_size=256):
    im = Image.open(img_path)
    im = np.array(im).astype(np.float64)
    if(len(im.shape) == 3):
        im = rgb2ycbcr(im/255)
        width, height, depth = im.shape
    else:
        width, height = im.shape
        depth = 1
    if (im_size == 'full'):
        return im, depth
    new_width = im_size
    new_height = im_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    if(depth == 3):
        img = np.array(im)[left:right, top:bottom, :].reshape((new_height, new_width, depth)).squeeze()
    else:
        img = np.array(im)[left:right, top:bottom].reshape((new_height, new_width)).squeeze()
    return img, depth

def read_image_resize(img_path, im_size=(256, 256)):
    im = Image.open(img_path)
    im = np.array(im).astype(np.float64)
    if(len(im.shape) == 3):
        im = rgb2ycbcr(im/255)
        width, height, depth = im.shape
    else:
        width, height = im.shape
        depth = 1
    # set new_width and new_height to the minimum size that allows to satisfy the ration im_size[0]/im_size[1]
    new_width = np.min((width, height))
    new_height = new_width
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    if(depth == 3):
        img = np.array(im)[left:right, top:bottom, :].reshape((new_height, new_width, depth)).squeeze()
        # linearly interpolate to the desired size
        # accounting for the fact that the image is in YCbCr
        new_img = np.zeros((im_size[0], im_size[1], depth))
        for i in range(3):
            new_img[:, :, i] = np.array(Image.fromarray(img[:, :, i]).resize((im_size[1], im_size[0]), Image.BICUBIC))
    else:
        img = np.array(im)[left:right, top:bottom].reshape((new_height, new_width)).squeeze()
        # linearly interpolate to the desired size
        new_img = np.array(Image.fromarray(img).resize((im_size), Image.BICUBIC))
    return new_img, depth

def read_image_resize_rect(img_path, im_size=(256, 256)):
    im = Image.open(img_path)
    new_height = im_size[0]
    new_width = im_size[1]
    resized_image = ImageOps.fit(im, (new_width, new_height), Image.BICUBIC)
    resized_image = np.array(resized_image).astype(np.float64)
    if (len(resized_image.shape) == 3):
        resized_image = rgb2ycbcr(resized_image/255)
        depth = resized_image.shape[2]
    else:
        depth = 1    
    resized_image = np.round(resized_image)
    return resized_image, depth

def read_image_rgb(img_path, im_size=256):
    im = Image.open(img_path)
    im = np.array(im).astype(np.float64)
    width, height, depth = im.shape
    new_width = im_size
    new_height = im_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = (width + new_width) // 2
    bottom = (height + new_height) // 2
    if(depth == 3):
        img = np.array(im)[left:right, top:bottom, :].reshape((new_height, new_width, depth)).squeeze()
    else:
        img = np.array(im)[left:right, top:bottom].reshape((new_height, new_width)).squeeze()
    return img, depth

# create a function that uses a 2D median filter to smooth a mask of 0-1 values
def smooth_mask(mask, radius):
    # create a copy of the mask
    smoothed_mask = np.copy(mask)
    # loop over the mask and smooth it
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            # get the current pixel value
            pixel = mask[i, j]
            # get the pixel window
            pixel_window = mask[max(0, i-radius):min(mask.shape[0], i+radius+1), max(0, j-radius):min(mask.shape[1], j+radius+1)]
            # get the number of pixels in the window
            num_pixels = pixel_window.shape[0] * pixel_window.shape[1]
            # get the number of pixels that are 1
            num_ones = np.sum(pixel_window)
            # if the number of ones is greater than half of the total number of pixels, set the pixel to 1
            if num_ones > num_pixels / 2:
                smoothed_mask[i, j] = 1
            # otherwise, set the pixel to 0
            else:
                smoothed_mask[i, j] = 0
    return smoothed_mask

def localvar(f):
    # Set default values
    # this default values are chosen based on scikit image implementation
    sigma = 1.5
    truncate = 3.5
    # Compute mean and mean of squared values
    ux = gaussian_filter(f, sigma=sigma, mode='reflect', truncate=truncate, cval=0)
    uxx = gaussian_filter(f**2, sigma=sigma, mode='reflect', truncate=truncate, cval=0)
    
    # Compute local variance
    vx = uxx - ux**2
    return vx


def crossvar(input1, input2):
    # Compute mean and mean of squared values
    sigma = 1.5
    truncate = 3.5
    # Compute mean and mean of squared values
    ux1 = gaussian_filter(input1, sigma=sigma, mode='reflect', truncate=truncate, cval=0)
    ux2 = gaussian_filter(input2, sigma=sigma, mode='reflect', truncate=truncate, cval=0)
    uxx = gaussian_filter(input1*input2, sigma=sigma, mode='reflect', truncate=truncate, cval=0)
    # Compute local variance
    vx = uxx - ux1*ux2
    return vx


#@njit
def blockize(X, bsize):
    # Convert a 2D image to 3D tensor based on a specified block size
    h = bsize[0]
    w = bsize[1]
    H, W = X.shape
    #create a 3D tensor with all the possible 8x8 blocks

    # Calculate number of blocks horizontally and vertically
    num_blocks_w = W // w
    num_blocks_h = H // h
    
    # Initialize empty tensor to store blocks
    Y = np.zeros((h*w, num_blocks_w * num_blocks_h))
    
    # Loop over each block and store in tensor
    for j in range(num_blocks_h):
        for i in range(num_blocks_w):
            x = i * h
            y = j * w
            block = X[x:x+h, y:y+w]
            Y[:, j*num_blocks_w+i] = np.ravel(block, 'F')
    return Y

def get_perceptual_q(f_train, quant, N):
    # weights for SSIM, from error model with sum(q) constraint. The closed
    # form is obtained from Lagrange multiplier trick
    c2 = (0.03*255)**2
    vx = localvar(f_train)
    gm = quant / np.sqrt(12*(2*vx + c2))
    qx = (gm.size + np.sum(np.power(gm, 2)))/np.sum(gm)*gm - np.power(gm, 2)
    vxv = blockize(qx, (N, N)) 
    return vxv, qx, vx

def get_perceptual_q_power(f_train, sigma2, N):
    # weights for SSIM, from error model with sum(q) constraint. The closed
    # form is obtained from Lagrange multiplier trick
    c2 = (0.03*255)**2
    vx = localvar(f_train)
    gm = np.sqrt(sigma2) / np.sqrt((2*vx + c2))
    qx = (gm.size + np.sum(np.power(gm, 2)))/np.sum(gm)*gm - np.power(gm, 2)
    vxv = blockize(qx, (N, N)) 
    return vxv, qx, vx

def get_perceptual_q_cross(f_train, f_compress, delta, samples_zi, N):
    # weights for SSIM, from error model with sum(q) constraint. The closed
    # form is obtained from Lagrange multiplier trick
    c2 = (0.03*255)**2
    vx = localvar(f_train)

    vm_fit = vonmises.fit(samples_zi[:, :].ravel('F')/(delta/2)*np.pi, fscale=1, loc=0)
    noise_power = ((delta/(2*np.pi))**2)*vonmises.stats(vm_fit[0], moments='v')

    sigma_xz = crossvar(f_train, f_compress)

    am = np.sqrt((2*sigma_xz+c2) / (2*vx + c2))/vx.size

    ev_0 = localvar(f_train - f_compress)

    ev_0 = 2*average_block(ev_0, N)

    print(np.min(ev_0), np.max(ev_0))

    gm = np.sqrt(np.abs(ev_0)) / np.sqrt((2*vx + c2))
    qx = (gm.size + np.sum(np.power(gm, 2)))/np.sum(am*gm)*am*gm - np.power(gm, 2)
    qx = np.clip(qx, 0.1, 1.9)
    qx = qx / np.sum(qx) * np.sum(np.ones_like(qx))
    vxv = blockize(qx, (N, N)) 
    return vxv, qx, vx, noise_power


def average_block(input, N):
    output = np.zeros_like(input)
    for j in range(input.shape[1]//N):
        for i in range(input.shape[0]//N):
            output[i*N:(i+1)*N, j*N:(j+1)*N] = np.mean(input[i*N:(i+1)*N, j*N:(j+1)*N])
    return output

def find_indices_perceptual(f_test, qqt, xcw):
    N = int(np.sqrt(xcw[0].shape[0]))
    qvec, _ , _ = get_perceptual_q(f_test, matlab_round(qqt[0, 0]), N)
    edmq = edmxy(xcw, qvec)
    idq = np.argmin(edmq, axis=0)
    return idq    

def get_q_roots(f_train,quant,lam=1):
    c2 = (0.03*255)**2
    rad = 3
    hwin = 11
    vx = localvar(f_train, rad, hwin)
    gm = quant / np.sqrt(12*(2*vx + c2))
    q = np.zeros_like(gm)
    lam = lam
    for i in range(gm.shape[0]):
        for j in range(gm.shape[1]):
            q[i, j] = np.max(np.real(np.roots([1, 2*gm[i, j], gm[i, j]**2, -1/lam])))
    return q

def edmxy(X, Y):
    # Euclidean distance matrix between X and Y
    n = X.shape[0]
    m = Y.shape[1]
    
    # Compute squared norms of X and Y
    dx = np.sum(X**2, axis=1)
    dy = np.sum(Y**2, axis=0)
    
    # Compute Euclidean distance matrix
    dmat = np.tile(dx, (m, 1)).T - 2 * np.dot(X, Y) + np.tile(dy, (n, 1))

    return dmat

def alt_apply_zig_zag(img, N=8):
    blk_out = np.zeros_like(img)
    zig_zag = zigzag(N)
    for i in range(N):
        for j in range(N):
            blk_out[i, j] = img[zig_zag[i, j] // N, zig_zag[i, j] % N]    
    return blk_out


def alt_inv_zig_zag(img, N=8):
    blk_out = np.zeros((N, N), dtype=img.dtype)
    zig_zag = zigzag(N)
    for i in range(N):
        for j in range(N):
            blk_out[zig_zag[i, j] // N, zig_zag[i, j] % N] = img[i, j]     
    return blk_out


def get_quantization_scales(nqs, sup_val=90, inf_val=12):
    # Quantization table
    qsnu = matlab_round(np.linspace(inf_val, sup_val, nqs))
    fac_qt = np.zeros_like(qsnu)
    fac_qt[qsnu > 50] = (100 - qsnu[qsnu > 50]) / 50
    fac_qt[qsnu <= 50] = 50 / qsnu[qsnu <= 50]
    return fac_qt, qsnu


def apply_zig_zag(img, N=8):
    zig_zag = zigzag(N)
    blk_out = np.zeros_like(img).ravel('F')
    blk_out[zig_zag.ravel('F')] = img.ravel('F')
    blk_out = blk_out.reshape(img.shape, order='F')
    return blk_out


def inv_zig_zag(img, N=8):
    vec_im = img.ravel('F')
    zig_zag = zigzag(N)
    blk_out = vec_im[zig_zag.ravel('F')].reshape(img.shape, order='F')
    return blk_out


def unifgrid(N):
    H = N
    W = N
    d1 = np.ones(H*W-1)
    d1[H-1::H] = 0
    dN = np.ones(H*W-H)
    W = (np.diag(d1, -1) + np.diag(d1, 1) +
          np.diag(dN, -H) + np.diag(dN, H))
    L = np.diag(np.sum(W, axis=1)) - W
    return L, W

def compute_basis(Lg, xcw, N):
    D = dct(np.eye(N), norm='ortho', axis=0)
    ddm = np.kron(D.T, D.T)
    Q = np.diag(xcw)
    eigvals, eigvecs = eigh(Lg, Q, eigvals_only=False)
    eigvals = np.real(eigvals)
    # sort eig_vals in descending order
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]    
    sg = np.sign(np.diag(eigvecs.T @ ddm))
    #inds = np.argsort(eigvals)
    U = eigvecs @ np.diag(sg)
    return U, eigvals

def kmeans_vq(ncw, vxv):
    bmax = np.ceil(np.log2(ncw))*2
    kmeans = KMeans(
        init="random",
        n_clusters=ncw,
        n_init=10,
        max_iter=300,
        random_state=42)
    kmeans.fit(vxv.T)
    xcw = kmeans.cluster_centers_        
    icw = kmeans.labels_
    p0, _ = np.histogram(icw, ncw)
    p0 = p0/len(icw)
    b0 = np.zeros(len(p0))
    b0[p0 == 0] = bmax
    b0[p0>0] = -np.log2(p0[p0>0])
    return xcw, b0, p0

def qgft(U, xcw, block):
    Q = np.diag(xcw)
    xvec = np.ravel(block, 'F')
    prod = U.T @ Q @ xvec
    out_block = np.reshape(prod, block.shape, 'F')
    return out_block

def iqgft(U, block):
    xvec = np.ravel(block, 'F')
    prod = U @ xvec
    out_block = np.reshape(prod, block.shape, 'F')
    return out_block

def sort_decreasing_probabilities(xcw, b0, p0):
    ids = np.argsort(p0)[::-1]
    p0 = p0[ids]
    b0 = b0[ids]
    xcw = xcw[ids, :]
    return xcw, b0, p0 


def matlab_round(x):
    return np.trunc(x+np.copysign(0.5,x))

def eight_connected_mtx(N):
    r = N
    c = N
    # Make the first diagonal vector (for horizontal connections)
    diagVec1 = np.tile(np.concatenate((np.ones(c-1), [0])), (r, 1))
    diagVec1 = np.ravel(diagVec1, 'C')
    diagVec1 = diagVec1[0:-1]
    # Make the second diagonal vector (for anti-diagonal connections)
    diagVec2 = np.hstack((0, diagVec1[:c*(r-1)]))
    # Make the third diagonal vector (for vertical connections)
    diagVec3 = np.ones(c*(r-1))
    # Make the fourth diagonal vector (for diagonal connections)
    diagVec4 = diagVec2[1:-1]
    # Construct the adjacency matrix using diagonals
    adj = np.diag(diagVec1, k=1) + np.diag(diagVec2, k=c-1) + np.diag(diagVec3, k=c) + np.diag(diagVec4, k=c+1)
    # Make the adjacency matrix symmetric by adding its transpose
    adj = adj + adj.T
    return adj


def directed_path(N):
    A = np.zeros((N, N))
    A[0:N-1, 1:N] = np.eye(N-1)
    return A


def non_directed_path(N):
    A = directed_path(N)
    A = A + A.T
    L = np.diag(np.sum(A, axis=1)) - A
    return L, A


def bdsnr(metric_set1, metric_set2):
  """
  BJONTEGAARD    Bjontegaard metric calculation
  Bjontegaard's metric allows to compute the average gain in psnr between two
  rate-distortion curves [1].
  rate1,psnr1 - RD points for curve 1
  rate2,psnr2 - RD points for curve 2
  returns the calculated Bjontegaard metric 'dsnr'
  code adapted from code written by : (c) 2010 Giuseppe Valenzise
  http://www.mathworks.com/matlabcentral/fileexchange/27798-bjontegaard-metric/content/bjontegaard.m
  """
  # pylint: disable=too-many-locals
  # numpy seems to do tricks with its exports.
  # pylint: disable=no-member
  # map() is recommended against.
  # pylint: disable=bad-builtin
  rate1 = [x[0] for x in metric_set1]
  psnr1 = [x[1] for x in metric_set1]
  rate2 = [x[0] for x in metric_set2]
  psnr2 = [x[1] for x in metric_set2]

  log_rate1 = map(math.log, rate1)
  log_rate2 = map(math.log, rate2)

  # Best cubic poly fit for graph represented by log_ratex, psrn_x.
  poly1 = np.polyfit(log_rate1, psnr1, 3)
  poly2 = np.polyfit(log_rate2, psnr2, 3)

  # Integration interval.
  min_int = max([min(log_rate1), min(log_rate2)])
  max_int = min([max(log_rate1), max(log_rate2)])

  # Integrate poly1, and poly2.
  p_int1 = np.polyint(poly1)
  p_int2 = np.polyint(poly2)

  # Calculate the integrated value over the interval we care about.
  int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
  int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

  # Calculate the average improvement.
  if max_int != min_int:
    avg_diff = (int2 - int1) / (max_int - min_int)
  else:
    avg_diff = 0.0
  return avg_diff


def bdrate(metric_set1, metric_set2):
    """
    BJONTEGAARD    Bjontegaard metric calculation
    Bjontegaard's metric allows to compute the average % saving in bitrate
    between two rate-distortion curves [1].
    rate1,psnr1 - RD points for curve 1
    rate2,psnr2 - RD points for curve 2
    adapted from code from: (c) 2010 Giuseppe Valenzise
    """
    # numpy plays games with its exported functions.
    # pylint: disable=no-member
    # pylint: disable=too-many-locals
    # pylint: disable=bad-builtin
    rate1 = [float(x[0]) for x in metric_set1]
    psnr1 = [float(x[1]) for x in metric_set1]
    rate2 = [float(x[0]) for x in metric_set2]
    psnr2 = [float(x[1]) for x in metric_set2]

    #breakpoint()

    log_rate1 = np.log(np.array(rate1))
    log_rate2 = np.log(np.array(rate2))
    #log_rate1 = map(math.log, rate1)
    #log_rate2 = map(math.log, rate2)

  # Best cubic poly fit for graph represented by log_ratex, psrn_x.
    try:
        poly1 = np.polyfit(psnr1, log_rate1, 3)
        poly2 = np.polyfit(psnr2, log_rate2, 3)
    except:
        return 100

    # Integration interval.
    min_int = max([min(psnr1), min(psnr2)])
    max_int = min([max(psnr1), max(psnr2)])

    # find integral
    p_int1 = np.polyint(poly1)
    p_int2 = np.polyint(poly2)

    # Calculate the integrated value over the interval we care about.
    int1 = np.polyval(p_int1, max_int) - np.polyval(p_int1, min_int)
    int2 = np.polyval(p_int2, max_int) - np.polyval(p_int2, min_int)

    # Calculate the average improvement.
    avg_exp_diff = (int2 - int1) / (max_int - min_int)

    # In really bad formed data the exponent can grow too large.
    # clamp it.
    if avg_exp_diff > 200:
        avg_exp_diff = 200

    # Convert to a percentage.
    avg_diff = (math.exp(avg_exp_diff) - 1) * 100

    return avg_diff


