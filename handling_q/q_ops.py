import numpy as np
from utils.coding_library import dpcm, rlgr
import scipy 
from utils.utils_lpit import reshape_image_blocks, invert_reshape_image_blocks 


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


class q_ops():

    def __init__(self, true_N, n_cwd, N, nqs=6):
        self.true_N = true_N
        self.n_cwd = n_cwd
        self.N = N
        self.nqs = nqs
        self.n_cwds = [2, 3, 4, 5, 6, 7, 8]

    def quantize_q(self, new_Q):
        n_cwds = self.n_cwds
        rdos = np.zeros((len(n_cwds)))
        centroids_list = []
        ind_closest_list = []
        ind_closest_420_list = []
        Q_list = []
        rate_list = []
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
            ind_closest, ind_closest_420, Q = self.quantize_q_cen(new_Q, centroids)
            centroids_list.append(centroids)
            ind_closest_list.append(ind_closest)
            ind_closest_420_list.append(ind_closest_420)
            Q_list.append(Q)
            SSE = np.sum(np.square((new_Q - Q)))
            rate = self.compress_q(ind_closest)
            rate_list.append(rate)
            lambda_ = 0.85*(2**3)
            rdos[indis] = SSE + lambda_*rate

        self.centroids_list = centroids_list
        self.ind_closest_list = ind_closest_list
        self.ind_closest_420_list = ind_closest_420_list
        self.Q_list = Q_list
        self.rdos = rdos
        self.n_cwds = n_cwds
        self.rate_list = rate_list

    def choose_ncwd(self):
        rdos = self.rdos
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


    def get_centroids(self, quant_level=None):
        output = (self.centroids)
        #for i in range(self.n_cwd):
        #    output[i, :] = np.ones_like(output[i, :])
        #output = np.sqrt(self.centroids)
        return output
    
    def compress_q(self, ind_closest):
        fdpcm = np.zeros_like(ind_closest)
        for j in range(ind_closest.shape[0]):
            tmp = ind_closest[j, :]
            tmp = tmp.reshape((len(tmp), 1))
            fdpcm[j, :] = dpcm(tmp, 1)[0].squeeze()
        top = fdpcm[:, 0]
        top = top.reshape((len(top), 1))
        fdpcm[:, 0] = dpcm(top, 1)[0].squeeze()
        bits = np.zeros((8))
        for l in range(8):
            bits[l] = len(rlgr(fdpcm.astype(np.int32), L = l+2))*8
        bits_1 = np.min(bits)+3

        # run the most likely
        bits_2 = 0
        array_unasigned = []
        for i in range(ind_closest.shape[0]):
            for j in range(ind_closest.shape[1]):
                if (i == 0 and j == 0):
                    most_likely = 0
                elif (i == 0):
                    # assign ind_closest if less than 10, or subtract 10 if more than 10
                    most_likely = ind_closest[i, j-1] 
                elif (j == 0):
                    most_likely = ind_closest[i-1, j]
                else:
                    tmp_1 = ind_closest[i-1, j] 
                    tmp_2 = ind_closest[i, j-1]
                    most_likely = np.minimum(tmp_1, tmp_2)
                if (ind_closest[i, j] == most_likely):
                    bits_2 += 1
                else:
                    bits_2 += 1
                    if (ind_closest[i, j] < most_likely):
                        array_unasigned.append(ind_closest[i, j])
                    else:
                        array_unasigned.append(ind_closest[i, j]-1)
        array_unasigned = np.asarray(array_unasigned)
        array_unasigned = array_unasigned.reshape((len(array_unasigned), 1))
        array_unasigned = dpcm(array_unasigned, 1)[0].squeeze()
        bits = np.zeros((8))
        for l in range(8):
            bits[l] = len(rlgr(array_unasigned.astype(np.int32), L = l+1))*8 + bits_2
        bits_2 = np.min(bits) + 3
        print(bits_1, bits_2)
        return np.min(np.array([bits_1, bits_2]))+1

    
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
        self.n_cwds = [2, 3, 4, 5, 6, 7, 8]

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
        self.n_cwds = [4, 5, 6, 7]

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
        self.n_cwds = [4, 5, 6, 7]

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