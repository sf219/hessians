import numpy as np
from utils.coding_library import dpcm, rlgr, jacenc, jdcenc
import scipy 
from utils.utils_lpit import reshape_image_blocks, invert_reshape_image_blocks, apply_zig_zag
from utils.bits_class import dpcm_smart
from scipy.spatial import ConvexHull
from video_codecs.utils_avc import enc_cavlc
import pillow_jpls
from PIL import Image
import os
import rans.rANSCoder as rANS
import zlib

def filter_q(Q, sigma=2):
    truncate = 3.5
    sigma = sigma
    r = int(truncate * sigma + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    
    pad = (win_size-1)//2
    #Q = Q.reshape(self.true_N[0], self.true_N[1], order='F')
    img1 = np.pad(Q, pad, mode='symmetric')

    x = np.linspace(-truncate, truncate, win_size)
    window = scipy.stats.norm.pdf(x, scale=sigma) * scipy.stats.norm.pdf(x[:, None], scale=sigma)
    window = window/np.sum(window)
    
    Q_filt = scipy.signal.convolve(img1, window, mode='valid')
    return Q_filt


def is_sorted_increasing(lst):
    return all(lst[i] <= lst[i + 1] for i in range(len(lst) - 1))

def is_sorted_decreasing(lst):
    return all(lst[i] >= lst[i + 1] for i in range(len(lst) - 1))

class q_ops():

    def __init__(self, true_N, n_cwd, N, nqs=6):
        self.true_N = true_N
        self.n_cwd = n_cwd
        self.N = N
        self.nqs = nqs
        self.n_cwds_or = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    def quantize_q(self, new_Q, img):
        n_cwds = self.n_cwds_or.copy()
        centroids_list = []
        ind_closest_list = []
        ind_closest_420_list = []
        Q_list = []
        rate_list = []
        SSE_list = []
        true_N = self.true_N
        N = self.N

        indis = -1
        for n_cwd in n_cwds:
            indis += 1
            name_target = self.name_target()
            str_load = 'data/centroids/centroids_' + name_target + '_' + str(n_cwd) + '_' + str(true_N) + '_' + str(N) + '.npy'
            try:
                centroids = np.load(str_load)
            except:
                try:
                    centroids = np.load('week_2/centroids/centroids_ssim_{}_{}.npy'.format(4, 256))
                except:
                    centroids = np.load('data/centroids/centroids_ssim_{}_{}_8.npy'.format(4, (256, 256)))
            ind_closest, ind_closest_420, Q_old = self.quantize_q_cen(new_Q, centroids)
            new_centroids, Q = self.scale_centroids(centroids, new_Q, ind_closest)
            centroids_list.append(new_centroids)
            ind_closest_list.append(ind_closest)
            ind_closest_420_list.append(ind_closest_420)
            Q_list.append(Q)
            SSE = np.sum(np.square(Q - new_Q))
            SSE_list.append(SSE)
            rate, _ = self.compress_q(ind_closest, n_cwd)
            rate_list.append(rate)

        while not is_sorted_increasing(rate_list):
            for i in range(1, len(rate_list)):
                if rate_list[i] < rate_list[i-1]:
                    del rate_list[i-1]
                    del SSE_list[i-1]
                    del centroids_list[i-1]
                    del ind_closest_list[i-1]
                    del ind_closest_420_list[i-1]
                    del Q_list[i-1]
                    del n_cwds[i-1]
                    break

        while not is_sorted_decreasing(SSE_list):
            for i in range(1, len(SSE_list)):
                if SSE_list[i] > SSE_list[i-1]:
                    del rate_list[i]
                    del SSE_list[i]
                    del centroids_list[i]
                    del ind_closest_list[i]
                    del ind_closest_420_list[i]
                    del Q_list[i]
                    del n_cwds[i]
                    break

        self.SSE_or = SSE_list[0]  #np.sum(np.square(new_Q - np.mean(new_Q)))
        self.rate_or = rate_list[0]  #np.mean(new_Q)

        # Convex hull throws an error if there are less than 3 points
        if len(rate_list) > 2:
            pairs = list(zip(rate_list, SSE_list))
            try:
                cv_hull = ConvexHull(pairs)
            except:
                cv_hull = ConvexHull(pairs, qhull_options='QJ')
                
            vertices = np.sort(cv_hull.vertices)

            rate_list = [rate_list[i] for i in vertices]
            SSE_list = [SSE_list[i] for i in vertices]
            centroids_list = [centroids_list[i] for i in vertices]
            ind_closest_list = [ind_closest_list[i] for i in vertices]
            ind_closest_420_list = [ind_closest_420_list[i] for i in vertices]
            n_cwds = [n_cwds[i] for i in vertices]
        

        self.centroids_list = centroids_list
        self.ind_closest_list = ind_closest_list
        self.ind_closest_420_list = ind_closest_420_list
        self.Q_list = Q_list
        self.rate_list = rate_list
        self.SSE_list = SSE_list
        self.n_cwds = n_cwds

    def choose_ncwd(self, lbr_mode=False):
        rdos = []
        SSE_or = self.SSE_or
        rate_or = self.rate_or

        lam = (self.SSE_list[-1] - SSE_or) / (rate_or - self.rate_list[-1])

        if (lbr_mode is True):
            lam = lam
        else:
            lam = lam

        for i in range(len(self.rate_list)):
            rdos.append(self.SSE_list[i] + lam*self.rate_list[i])
            print('n_cwd: ', self.n_cwds[i], 'rate: ', self.rate_list[i], 'SSE: ', self.SSE_list[i], 'RDO: ', rdos[i])

        ind = np.argmin(rdos)
        self.n_cwd = self.n_cwds[ind]
        self.ind_closest = self.ind_closest_list[ind]
        self.ind_closest_420 = self.ind_closest_420_list[ind]
        self.Q = self.Q_list[ind]
        self.centroids = self.centroids_list[ind]
        self.overhead_bits = self.rate_list[ind]

    def quantize_q_cen(self, new_Q, centroids):
        n_cwd = centroids.shape[0]
        N = self.N
        Q = np.zeros((self.true_N[0], self.true_N[1]))
        ind_closest = np.zeros((self.true_N[0]//N, self.true_N[1]//N))
        for i in range(0, self.true_N[0], N):
            for j in range(0, self.true_N[1], N):
                new_Q_blk = new_Q[i:i+N, j:j+N].ravel('F')
                dists = np.zeros(n_cwd)
                for k in range(n_cwd):
                    dists[k] = np.linalg.norm(np.array(new_Q_blk) - 
                                              centroids[k, :].ravel('F'))
                ind_closest[i//N, j//N] = np.argmin(dists)
                Q[i:i+N, j:j+N] = centroids[np.argmin(dists), :]
        ind_closest_420 = np.zeros((ind_closest.shape[0]//2, ind_closest.shape[1]//2))
        # iterate in blocks of size 2 by 2
        for i in range(ind_closest.shape[0]//2):
            for j in range(ind_closest.shape[1]//2):
                tmp = ind_closest[2*i:2*i+2, 2*j:2*j+2]
                tmp = tmp.ravel('F')
                mean_tmp = np.round(np.mean(tmp))
                ind_closest_420[i, j] = int(mean_tmp)
        return ind_closest, ind_closest_420, Q


    def scale_centroids(self, centroids, new_Q, ind_closest):
        #iterate over all the blocks of new_Q
        n_cwd = centroids.shape[0]
        N = self.N
        true_N = self.true_N
        avg_weight = np.zeros((n_cwd))
        counts = np.zeros((n_cwd))
        for i in range(0, true_N[0], N):
            for j in range(0, true_N[1], N):
                new_Q_blk = new_Q[i:i+N, j:j+N].ravel('F')
                ind = ind_closest[i//N, j//N].astype(int)
                avg_weight[ind] += np.mean(new_Q_blk)
                counts[ind] += 1
        avg_weight = avg_weight/counts
        avg_weight[np.isnan(avg_weight)] = 1
        ratio = np.zeros_like(avg_weight)
        for i in range(n_cwd):
            ratio[i] = avg_weight[i]/np.mean(centroids[i, :])
        #print('ratio: ', ratio)
        new_centroids = np.zeros(centroids.shape)
        for i in range(n_cwd):
            new_centroids[i, :] = centroids[i, :]*ratio[i]
        
        Q = np.zeros_like(new_Q)
        for i in range(0, true_N[0], N):
            for j in range(0, true_N[1], N):
                Q[i:i+N, j:j+N] = new_centroids[ind_closest[i//N, j//N].astype(int), :]

        normalizer = Q.size / np.sum(Q)

        Q = Q * normalizer
        
        for i in range(n_cwd):
            new_centroids[i, :] = new_centroids[i, :] * normalizer

        return new_centroids, Q

    def get_centroids(self, quant_level=None):
        output = (self.centroids)
        return output
    
    def compress_q(self, ind_closest, n_cwd=None):
        ind_closest = ind_closest.copy()
        
        unique = np.unique(ind_closest).astype(int)
        probs = np.zeros((unique.shape[0]))
        for i in range(unique.shape[0]):
            probs[i] = np.mean(ind_closest == unique[i])
        or_ind_closest = ind_closest.copy()
        arg_inc = np.argsort(probs)[::-1]
        for i in range(unique.shape[0]):
            ind_closest[or_ind_closest == unique[arg_inc[i]]] = i

        fdpcm = dpcm_smart(ind_closest)
        bits = np.zeros((8))
        for l in range(8):
            bits[l] = len(rlgr(fdpcm.ravel('F').astype(np.int32), L=l+1))*8
        index_min = np.argmin(bits)
        byte_seq = rlgr(fdpcm.astype(np.int32), L=index_min+1)
        bits_1 = np.min(bits)+3
        bits_2 = 0

        fdpcm_save = fdpcm.copy()
        fdpcm_save[fdpcm_save<0] = (n_cwd) - fdpcm_save[fdpcm_save<0]
        im = Image.fromarray(ind_closest.astype(np.uint8))
        im.save('fdpcm.webp', lossless=True, quality=100)
        read_im = Image.open('fdpcm.webp')
        read_arr = np.array(read_im)[:, :, 0].astype(np.int32)
        #mod_arr = read_arr.copy().astype(np.int32)
        #mod_arr[mod_arr >= n_cwd] = -(mod_arr[mod_arr>=n_cwd] - (n_cwd))
        bits_2 = os.stat('fdpcm.webp').st_size*8
        #print('bits_2: ', bits_2, 'bits_1: ', bits_1, 'fdpcm: ', np.sum(np.abs(read_arr - ind_closest)))
        bits = np.minimum(bits_1, bits_2)+1
        return bits, byte_seq

    def uncompress_q(self, byte_seq):
        pass

    def name_target(self):
        return 'mse'

    def normalize_q(self, Q):
        Q = Q.squeeze()
        #Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(Q) * Q.size
        return Q


class q_ops_ssim(q_ops):

    def __init__(self, true_N, n_cwd, N, nqs):
        super().__init__(true_N, n_cwd, N, nqs)

    def normalize_q(self, Q):
        Q = Q.squeeze()
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q

    def name_target(self):
        return 'ssim'
    

class q_ops_msssim(q_ops):
    
    def __init__(self, true_N, n_cwd, N, nqs):
        super().__init__(true_N, n_cwd, N, nqs)


    def normalize_q(self, Q):
        Q = Q.squeeze()
        Q[Q < 0] = 0
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        medianQ = np.median(Q)
        Q[Q > 5*medianQ] = 5*medianQ
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q

    def name_target(self):
        return 'msssim'


class q_ops_brisque(q_ops):
    
    def __init__(self, true_N, n_cwd, N, nqs):
        super().__init__(true_N, n_cwd, N, nqs)

    def name_target(self):
        return 'brisque'
    
    def normalize_q(self, Q):
        #Q = self.filter_Q(Q)
        Q = Q.squeeze()
        Q[Q < 0] = 0
        #Q = filter_q(Q, 1)
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        medianQ = np.median(Q)
        Q[Q > 3*medianQ] = 3*medianQ
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q
    

class q_ops_niqe(q_ops):
    
    def __init__(self, true_N, n_cwd, N, nqs):
        super().__init__(true_N, n_cwd, N, nqs)

    def name_target(self):
        return 'brisque'
    
    def normalize_q(self, Q, sigma=1):
        #Q = self.filter_Q(Q)
        Q[np.isnan(Q)] = np.median(Q)
        Q = Q.squeeze()
        Q[Q < 0] = 0
        #Q = filter_q(Q, sigma)
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        medianQ = np.median(Q)
        Q[Q > 3*medianQ] = 3*medianQ
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q


class q_ops_lpips(q_ops):
    
    def __init__(self, true_N, n_cwd, N, nqs):
        super().__init__(true_N, n_cwd, N, nqs)

    def name_target(self):
        return 'lpips'
    
    def normalize_q(self, Q, sigma=0.1):
        Q = Q.squeeze()
        Q[Q < 0] = 0
        #Q = filter_q(Q, sigma)
        Q = Q + 0.01*np.max(Q)
        #Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        medianQ = np.median(Q)
        Q[Q > 5*medianQ] = 5*medianQ
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q
    

class q_ops_resnet(q_ops):

    def __init__(self, true_N, n_cwd, N, nqs):
        super().__init__(true_N, n_cwd, N, nqs)

    def name_target(self):
        return 'resnet'
    
    def normalize_q(self, Q):
        Q = Q.squeeze()
        Q[Q < 0] = 0
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        medianQ = np.median(Q)
        Q[Q > 5*medianQ] = 5*medianQ
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q