import torch
import numpy as np
from utils.utils_lpit import read_image_resize
from scipy.linalg import hadamard
from utils.coding_library import dpcm, rlgr
import scipy
from utils.q_utils import ssim_func, ms_ssim_func, haarpsi_func, brisque_func, compute_LPIPS_gs, clip_func

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class compute_Q():
    def __init__(self, true_N, n_cwd, N=8):
        self.true_N = true_N
        self.n_cwd = n_cwd
        self.N = N
        self.sampling_depth = 128

        name_target = self.name_target()
        str_load = 'data/centroids/centroids_' + name_target + '_' + str(n_cwd) + '_' + str(true_N) + '_' + str(N) + '.npy'
        try:
            self.centroids = np.load(str_load)
        except:
            try:
                self.centroids = np.load('week_2/centroids/centroids_ssim_{}_{}.npy'.format(4, 256))
            except:
                print('Centroids not found')
        self.set_samplers()
        self.master_func = lambda x, y: torch.mean(torch.square(x - y))

    def set_samplers(self):
        # values needed to approximate the Hessian
        self.gauss_sample = np.random.randn(self.true_N[0], self.true_N[1],
                                             self.sampling_depth)
        # values to compute the Hadamard approximation
        had = hadamard(self.sampling_depth).T
        # repeat the columns of the matrix had until we have a matrix of size true_N*true_N, sampling_depth
        had_tmp = np.repeat(had, (self.true_N[0]*self.true_N[1])//self.sampling_depth, 1).T
        self.had = np.zeros_like(self.gauss_sample)
        self.normalizer = np.zeros((self.true_N[0], self.true_N[1]))
        for i in range(self.sampling_depth):
            self.had[:, :, i] = had_tmp[:, i].reshape(self.true_N[0], self.true_N[1])
            self.normalizer += np.array(self.had[:, :, i])**2
        self.normalizer = self.normalizer.reshape(self.true_N[0]*self.true_N[1],)
        self.had = np.array(self.had)
        self.gauss_sample = np.array(self.gauss_sample)
        self.rad = np.sign(np.random.randn(self.true_N[0], self.true_N[1], self.sampling_depth))

    def compute_Q(self, img, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x)

        def faster_hessian_diagonal(image):
            num = image.size
            shape = image.shape
            s1 = shape[0]
            s2 = shape[1]

            def fun_eval(x):
                hvp = hvp_faster(np.eye(num)[:, x].reshape(s1, s2))
                return hvp.reshape(num, 1)[x]

            # vectorization is not really an option here
            # it takes more time to compute that doing the norm trick 
            second_Q = np.zeros((num,))
            for j in range(num):
                second_Q[j] = fun_eval(j)
            return second_Q
        
        Q = faster_hessian_diagonal(img)
        Q = self.normalize_Q(Q)
        return Q
    
    def prenormalize_img(self, img):
        img = 2/255*img
        return img
    
    def compute_Q_ip(self, img, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x)
        num = img.size

        def fun_eval(i):
            rand_vec = self.sampler[:, :, i].squeeze()
            hvp = hvp_faster(rand_vec)
            return hvp.reshape(num,)

        Q = np.zeros((num,))
        for j in range(self.sampling_depth):
            Q += np.square(fun_eval(j))
        Q = np.sqrt(Q)
        Q = self.normalize_Q(Q)
        return Q

    def sample_q_ip_gaussian_train(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.gauss_sample
        Q = self.compute_Q_ip(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        return Q
  
    def sample_q_ip_radamacher_train(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.rad
        Q = self.compute_Q_ip(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        return Q
    
    def compute_Q_diag(self, img, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x)
        num = img.size

        def fun_eval(i):
            rand_vec = self.sampler[:, :, i].squeeze()
            hvp = hvp_faster(rand_vec)
            out = (hvp*rand_vec).reshape(num,)
            return out

        Q = np.zeros((num,))
        normalizer = np.zeros((num,))
        for j in range(self.sampling_depth):
            Q += fun_eval(j)
            rand_vec = self.sampler[:, :, j].squeeze()
            normalizer += np.square(rand_vec).reshape(num,)
        Q = Q / normalizer
        Q = self.normalize_Q(Q)
        return Q


    def sample_q_diag_gaussian_train(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.gauss_sample
        Q = self.compute_Q_diag(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        return Q
    
    def sample_q_diag_radamacher_train(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.rad
        Q = self.compute_Q_diag(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        return Q
        
    def sample_q_diag_hadamard_train(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.had
        Q = self.compute_Q_diag(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        return Q
    
    def compute_Q_det(self, img, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x)
        num = img.shape

        def fun_eval(i, j):
            rand_vec = np.zeros(num)
            rand_vec[i, j] = 1
            hvp = hvp_faster(rand_vec)
            out = (hvp*rand_vec).reshape(num)
            return out

        Q = np.zeros((num))
        for j in range(num[0]):
            for i in range(num[1]):
                Q += fun_eval(j, i)
        Q = self.normalize_Q(Q)
        return Q
    
    def quantize_q(self, new_Q, level=None):
        N = self.N
        Q = np.zeros((self.true_N[0], self.true_N[1]))
        ind_closest = np.zeros((self.true_N[0]//N, self.true_N[1]//N))
        for i in range(0, self.true_N[0], N):
            for j in range(0, self.true_N[1], N):
                new_Q_blk = new_Q[i:i+N, j:j+N].ravel('F')
                dists = np.zeros(self.n_cwd)
                for k in range(self.n_cwd):
                    dists[k] = np.linalg.norm(np.array(new_Q_blk) - 
                                              self.centroids[k, :].ravel('F'))
                ind_closest[i//N, j//N] = np.argmin(dists)
                Q[i:i+N, j:j+N] = self.centroids[np.argmin(dists), :]
        return ind_closest, Q
    
    def get_centroids(self, quant_level=None):
        output = (self.centroids)
        #for i in range(self.n_cwd):
        #    output[i, :] = np.ones_like(output[i, :])
        #output = np.sqrt(self.centroids)
        return output
    
    def compress_Q(self, ind_closest):
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

    def normalize_Q(self, Q):
        Q = Q.squeeze()
        #Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(Q) * Q.size
        return Q

    def filter_Q(self, Q):
        truncate = 3.5
        sigma = 6
        r = int(truncate * sigma + 0.5)  # radius as in ndimage
        win_size = 2 * r + 1
        
        pad = (win_size-1)//2
        Q = Q.reshape(self.true_N[0], self.true_N[1], order='F')
        img1 = np.pad(Q, pad, mode='symmetric')

        x = np.linspace(-truncate, truncate, win_size)
        window = scipy.stats.norm.pdf(x, scale=sigma) * scipy.stats.norm.pdf(x[:, None], scale=sigma)
        window = window/np.sum(window)
        
        Q_filt = scipy.signal.convolve(img1, window, mode='valid')
        return Q_filt.ravel('F')

    def name_target(self):
        return 'mse'

    def compute_ood(self, img, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x)

        def fun_eval(i):
            rand_vec = self.rad[:, :, i].squeeze()
            hvp = hvp_faster(rand_vec)
            hvp = np.sum(hvp * rand_vec)

            rand_vec = self.gauss_sample[:, :, i].squeeze() 
            hvp_den = hvp_faster(rand_vec)
            hvp_den = np.sum(hvp_den * rand_vec)
            return hvp, hvp_den

        samples = np.zeros((self.sampling_depth))
        sampling_den = np.zeros((self.sampling_depth))
        for j in range(self.sampling_depth):
            samples[j], sampling_den[j] = fun_eval(j)
        ood = np.var(samples)
        ood_den = np.var(sampling_den)
        output = np.minimum(ood/ood_den, 1)
        return output

    def sample_q(self, img, level=None):
        Q = self.sample_q_train(img)
        ind_closest, quant_Q = self.quantize_q(Q, level)
        return Q, ind_closest, quant_Q
    
    def sample_q_train(self, img, level=None):
        Q = self.sample_q_ip_radamacher_train(img)
        return Q

    def sample_ood(self, img, level=None):
        small_img = self.prenormalize_img(img)
        ood = self.compute_ood(small_img)
        return ood

    def sample_q_det(self, img, level=None):
        small_img = self.prenormalize_img(img)
        Q = self.compute_Q_det(small_img)
        return Q

    def check_first_derivative(self, img):
        img = self.prenormalize_img(img)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img = img.type(torch.float32)
        img.requires_grad = True
        func = lambda x: self.master_func(img, x)
        img2 = torch.ones_like(img)
        handle_out = torch.autograd.functional.jvp(func, img, img2)[1]
        del img, img2
        torch.cuda.empty_cache()
        out_val = handle_out.cpu().numpy().squeeze()
        print(self.name_target() + ' ' + str(out_val))

    def check_evaluation(self, img):
        img = self.prenormalize_img(img)
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
        img = img.type(torch.float32)
        func = lambda x: self.master_func(img, x)
        handle_out = func(img)
        del img
        torch.cuda.empty_cache()
        out_val = handle_out.cpu().detach().numpy().squeeze()
        print(self.name_target() + ' ' + str(out_val))

    def sample_hessian(self, img, sampler):
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device).type(torch.float32)
        sampler = torch.from_numpy(sampler).unsqueeze(0).unsqueeze(0).to(device).type(torch.float32)
        func = lambda x: self.master_func(img, x)
        handle_out = torch.autograd.functional.hvp(func, img, sampler)[1]
        del img, sampler
        torch.cuda.empty_cache()
        out_val = handle_out.cpu().detach().numpy().squeeze()
        return out_val
    

class compute_Q_lpips(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    # to use the LPIPS metric
    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)        
        self.master_func = lambda x, y: compute_LPIPS_gs(x, y)

    def prenormalize_img(self, img):
        return super().prenormalize_img(img)-1

    def normalize_Q(self, Q):
        #Q = self.filter_Q(Q)
        #Q = Q + 0.01*np.max(Q)
        Q[Q < 0] = 0
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q
    
    def name_target(self):
        return 'lpips'


class compute_Q_ssim(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    # to use the SSIM metric
    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: ssim_func(x, y)

    def name_target(self):
        return 'ssim'


class compute_Q_msssim(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    # to use the MS-SSIM metric
    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: ms_ssim_func(x, y)

    def name_target(self):
        return 'msssim'


class compute_Q_haarpsi(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    # to use the MS-SSIM metric
    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: haarpsi_func(x, y)

    def name_target(self):
        return 'lpips'


class compute_Q_brisque(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    # to use the MS-SSIM metric
    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: brisque_func(x, y)

    def normalize_Q(self, Q):
        #Q = self.filter_Q(Q)
        Q[Q < 0] = 0
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(np.abs(Q)) * Q.size
        return Q

    def sample_q_train(self, img, level=None):
        Q = self.sample_q_ip_radamacher_train(img)
        return Q

    def name_target(self):
        return 'brisque'
    

class compute_Q_clip(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    # to use the MS-SSIM metric
    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)

    def normalize_Q(self, Q):
        Q = self.filter_Q(Q)
        Q = Q + 0.01*np.max(Q)
        Q = Q / np.sum(Q) * Q.size
        return Q

    def name_target(self):
        return 'clip'