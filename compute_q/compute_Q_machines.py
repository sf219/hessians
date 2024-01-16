from compute_q.compute_Q_pytorch import compute_Q
import torch
import numpy as np
from scipy.linalg import hadamard
import scipy 
from utils.q_utils import resnet_loss, resnet_loss_layer, vgg_loss_layer, alexnet_loss_layer
from scipy.linalg import hadamard
from utils.q_utils import mnist_loss_layer

#model_conv = torch.load('sketch/models/trained_insects.pth')

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def two_inp_jvp_resnet18(img1, img2):
    img1 = torch.from_numpy(img1).unsqueeze(0).to(device).float()
    img2 = torch.from_numpy(img2).unsqueeze(0).to(device).float()
    func = lambda x: resnet_loss(img1, x)
    handle_out = torch.autograd.functional.hvp(func, img1, img2)[1]
    del img1, img2
    torch.cuda.empty_cache()
    handle_out = handle_out.cpu().numpy().squeeze()
    return handle_out


def two_inp_jvp_mse(img1, img2):
    img1 = torch.from_numpy(img1).unsqueeze(0).to(device).float()
    img2 = torch.from_numpy(img2).unsqueeze(0).to(device).float()
    func = lambda x: torch.mean(torch.square(img1 - x))
    handle_out = torch.autograd.functional.hvp(func, img1, img2)[1]
    del img1, img2
    torch.cuda.empty_cache()
    handle_out = handle_out.cpu().numpy().squeeze()
    return handle_out

class compute_Q_machines(compute_Q):
        
    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)        
        self.sampling_depth = 224
        self.handle_fun = lambda x, y: two_inp_jvp_resnet18(x, y)

    def set_samplers(self):
        # values needed to approximate the Hessian
        self.gauss_sample = np.random.randn(3, self.true_N[0], self.true_N[1],
                                                self.sampling_depth)
        # values to compute the Hadamard approximation
        self.gauss_sample = np.array(self.gauss_sample)
        had = hadamard(self.sampling_depth).T
        # repeat the columns of the matrix had until we have a matrix of size true_N*true_N, sampling_depth
        had_tmp = np.repeat(had, (self.true_N[0]*self.true_N[1])//self.sampling_depth, 1).T
        self.had = np.zeros((self.true_N[0], self.true_N[1], self.sampling_depth))
        self.normalizer = np.zeros((self.true_N[0], self.true_N[1]))
        for i in range(self.sampling_depth):
            self.had[:, :, i] = had_tmp[:, i].reshape(self.true_N[0], self.true_N[1])
            self.normalizer += np.array(self.had[:, :, i])**2
        self.normalizer = self.normalizer.reshape(self.true_N[0], self.true_N[1], 1)
        self.had = self.had.reshape(1, self.true_N[0], self.true_N[1], self.sampling_depth)
        self.normalizer = np.tile(self.normalizer, (1, 1, 3))
        self.had = np.tile(self.had, (3, 1, 1, 1))
        self.had = np.array(self.had)
        self.rad = np.sign(np.random.randn(3, self.true_N[0], self.true_N[1], self.sampling_depth))

    def compute_Q_ip(self, img, level=None):
        hvp_faster = lambda x: self.handle_fun(img, x)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, i].squeeze()
            hvp = hvp_faster(rand_vec)
            hvp = hvp.transpose((1, 2, 0))
            return hvp

        Q = np.zeros((num[1], num[2], num[0]))
        for j in range(self.sampling_depth):
            Q += np.square(fun_eval(j))
        Q = np.sqrt(Q)
        Q_in = np.sqrt(np.mean(Q**2, axis=2))
        Q = Q_in.ravel('F')
        Q = self.normalize_Q(Q)
        Q = Q.reshape((num[1], num[2]), order='F')
        return Q
    
    def compute_Q_diag(self, img, level=None):
        hvp_faster = lambda x: self.handle_fun(img, x)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, i].squeeze()
            hvp = hvp_faster(rand_vec)
            hvp = hvp.transpose((1, 2, 0))
            hvp_den = (rand_vec*rand_vec).transpose((1, 2, 0))
            return hvp, hvp_den
        
        Q = np.zeros((num[1], num[2], num[0]))
        normalizer = np.zeros((num[1], num[2], num[0]))
        for j in range(self.sampling_depth):
            Q_tmp, normalizer_tmp = fun_eval(j)
            Q += Q_tmp
            normalizer += normalizer_tmp
        Q = Q / normalizer
        Q_in = np.sqrt(np.mean(Q**2, axis=2))
        Q = Q_in.ravel('F')
        Q = self.normalize_Q(Q)
        Q = Q.reshape((num[1], num[2]), order='F')
        return Q

    def normalize_Q(self, Q):
        Q = Q.ravel('F')
        Q = self.filter_Q(Q)
        Q = Q.reshape((self.true_N[0], self.true_N[1]), order='F')
        Q = Q / np.sum(Q) * Q.size
        return Q
        
    def name_target(self):
        return 'resnet18'

    def sample_q(self, img, level=None):
        Q = self.sample_q_train(img)
        ind_closest, quant_Q = self.quantize_q(Q, level)
        return Q, ind_closest, quant_Q

    def sample_q_train(self, img, level=None):
        self.sampler = self.gauss_sample
        Q = self.compute_Q_ip(img)
        return Q

    def filter_Q(self, Q):
        truncate = 3.5
        sigma = 2
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

    def get_centroids(self, quant_level=None):
        output = (self.centroids)
        return output


class compute_Q_layer(compute_Q_machines):

    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)        
        self.sampling_depth = 128
        self.master_func = lambda x, y, layer: resnet_loss_layer(x, y, layer)

    def compute_ood(self, img, layer, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x, layer)

        def fun_eval(i):
            rand_vec = self.rad[:, :, :, i]
            hvp = hvp_faster(rand_vec)
            rand_vec = rand_vec.squeeze()
            hvp = np.sum(hvp * rand_vec)

            rand_vec = self.gauss_sample[:, :, :, i]
            hvp_den = hvp_faster(rand_vec)
            rand_vec = rand_vec.squeeze()
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

    def compute_Q_ip(self, img, layer, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x, layer)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, i]
            hvp = hvp_faster(rand_vec)
            hvp = self.transform_after_sampling(hvp)
            return hvp

        Q = np.zeros((num[1], num[2], num[0]))
        for j in range(self.sampling_depth):
            Q += np.square(fun_eval(j))
        Q = np.sqrt(Q)
        Q_in = self.color_to_gray(Q)
        Q = Q_in.ravel('F')
        Q = self.normalize_Q(Q)
        Q = Q.reshape((num[1], num[2]), order='F')
        return Q

    def compute_Q_diag(self, img, layer, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x, layer)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, i]
            hvp = hvp_faster(rand_vec)
            rand_vec = rand_vec.squeeze()
            hvp = hvp*rand_vec
            hvp = self.transform_after_sampling(hvp)
            hvp_den = self.transform_after_sampling(rand_vec*rand_vec)
            return hvp, hvp_den
        
        Q = np.zeros((num[1], num[2], num[0]))
        normalizer = np.zeros((num[1], num[2], num[0]))
        for j in range(self.sampling_depth):
            Q_tmp, normalizer_tmp = fun_eval(j)
            Q += Q_tmp
            normalizer += normalizer_tmp
        Q = Q / normalizer
        Q_in = self.color_to_gray(Q)
        Q = Q_in.ravel('F')
        Q = self.normalize_Q(Q)
        Q = Q.reshape((num[1], num[2]), order='F')
        return Q


    def sample_q_train_ip(self, img, layer, level=None):
        self.sampler = self.gauss_sample
        Q = self.compute_Q_ip(img, layer)
        return Q
    
    def sample_q_train_diag(self, img, layer, level=None):
        self.sampler = self.gauss_sample
        Q = self.compute_Q_diag(img, layer)
        return Q

    def sample_hessian(self, img, sampler, layer):
        img = torch.from_numpy(img).unsqueeze(0).to(device).float()
        sampler = torch.from_numpy(sampler).unsqueeze(0).to(device).float()
        func = lambda x: self.master_func(img, x, layer)
        handle_out = torch.autograd.functional.hvp(func, img, sampler)[1]
        del img, sampler
        torch.cuda.empty_cache()
        handle_out = handle_out.cpu().numpy().squeeze()
        return handle_out
    

    def transform_after_sampling(self, Q):
        return Q.transpose((1, 2, 0))

    def color_to_gray(self, Q):
        return np.sqrt(np.mean(Q**2, axis=2))
    
class compute_Q_layer_vgg16(compute_Q_layer):

    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)        
        self.sampling_depth = 128
        self.master_func = lambda x, y, layer: vgg_loss_layer(x, y, layer)


class compute_Q_layer_alex(compute_Q_layer):

    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)        
        self.sampling_depth = 128
        self.master_func = lambda x, y, layer: alexnet_loss_layer(x, y, layer)


class compute_Q_layer_mnist(compute_Q_layer):

    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)        
        self.sampling_depth = 256
        self.master_func = lambda x, y, layer: mnist_loss_layer(x, y, layer)
        self.set_samplers()


    def set_samplers(self):
        # values needed to approximate the Hessian
        self.gauss_sample = np.random.randn(1, self.true_N[0], self.true_N[1],
                                                self.sampling_depth)
        # values to compute the Hadamard approximation
        self.gauss_sample = np.array(self.gauss_sample)
        self.rad = np.sign(np.random.randn(1, self.true_N[0], self.true_N[1], self.sampling_depth))


    def transform_after_sampling(self, Q):
        Q = np.reshape(Q, (self.true_N[0], self.true_N[1], 1))
        return Q

    def color_to_gray(self, Q):
        return Q.squeeze()

    def normalize_Q(self, Q):
        return Q / np.sum(Q) * Q.size
    
    def compute_exact_hessian(self, img, layer, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x, layer)
        N = 28

        def fun_eval(i):
            rand_vec = np.zeros((N*N))
            rand_vec[i] = 1
            rand_vec = rand_vec.reshape(1, N, N)
            hvp = hvp_faster(rand_vec)
            hvp = hvp.ravel()
            return hvp
        
        H = np.zeros((N*N, N*N))
        for i in range(N*N):
            H[:, i] = fun_eval(i)
        return H
