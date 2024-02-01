from compute_q.compute_Q_pytorch import compute_Q
import torch
import numpy as np
from scipy.linalg import hadamard
import scipy 
from utils.q_utils import resnet_loss, resnet_loss_layer, vgg_loss_layer, alexnet_loss_layer, compute_LPIPS_gs, jacobian_loss
from scipy.linalg import hadamard
from utils.q_utils import mnist_loss_layer
from torchvision import transforms

#model_conv = torch.load('sketch/models/trained_insects.pth')

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

class compute_Q_machines(compute_Q):
        
    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)        
        self.sampling_depth = 128
        self.master_func = lambda x, y: compute_LPIPS_gs(x, y)
        self.jacobian_loss = lambda x: jacobian_loss(x)

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

    def color_to_gray(self, Q):
        weights = np.array([65.481, 128.553, 24.966])/255
        Q = weights[0]*Q[0, :, :] + weights[1]*Q[1, :, :] + weights[2]*Q[2, :, :]
        return Q

    def compute_Q_ip(self, img, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, i].squeeze()
            hvp = hvp_faster(rand_vec)
            return hvp

        Q = np.zeros(num)
        for j in range(self.sampling_depth):
            Q += np.square(fun_eval(j))
        Q = np.sqrt(Q)
        Q_in = self.color_to_gray(Q)
        Q = self.normalize_Q(Q_in)
        return Q
    
    def compute_Q_diag(self, img, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, i].squeeze()
            hvp = hvp_faster(rand_vec)*rand_vec
            hvp_den = (rand_vec*rand_vec)
            return hvp, hvp_den
        
        Q = np.zeros(num)
        normalizer = np.zeros(num)
        for j in range(self.sampling_depth):
            Q_tmp, normalizer_tmp = fun_eval(j)
            Q += Q_tmp
            normalizer += normalizer_tmp
        Q = Q / normalizer
        Q_in = self.color_to_gray(Q)
        Q = self.normalize_Q(Q_in)
        return Q

    def normalize_Q(self, Q):
        Q = Q.ravel('F')
        Q = self.filter_Q(Q)
        Q = Q.reshape((self.true_N[0], self.true_N[1]), order='F')
        Q = Q / np.sum(Q) * Q.size
        return Q
        
    def name_target(self):
        return 'resnet18'

    def sample_jac_ip(self, img, level=None):
        Q = self.sample_jac_train_ip(img)
        ind_closest, quant_Q = self.quantize_q(Q, level)
        return Q, ind_closest, quant_Q
    
    def sample_jac_train_ip(self, img, level=None):
        self.sampler = self.gauss_sample
        Q = self.compute_Q_ip_jacobian(img)
        return Q

    def sample_q_ip(self, img, level=None):
        Q = self.sample_q_train_ip(img)
        ind_closest, quant_Q = self.quantize_q(Q, level)
        return Q, ind_closest, quant_Q

    def sample_q_train_ip(self, img, level=None):
        self.sampler = self.gauss_sample
        Q = self.compute_Q_ip(img)
        return Q
    
    def sample_q_diag(self, img, level=None):
        Q = self.sample_q_train_diag(img)
        ind_closest, quant_Q = self.quantize_q(Q, level)
        return Q, ind_closest, quant_Q

    def sample_q_train_diag(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.gauss_sample
        Q = self.compute_Q_diag(small_img)
        return Q

    def sample_q(self, img, level=None):
        Q = self.sample_q_train(img)
        ind_closest, quant_Q = self.quantize_q(Q, level)
        return Q, ind_closest, quant_Q
    
    def sample_q_train(self, img, level=None):
        Q = self.sample_q_ip_radamacher_train(img)
        return Q

    def sample_q_ip_radamacher_train(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.rad
        Q = self.compute_Q_ip(small_img)
        return Q
    
    def sample_q_ip_hadamard_train(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.had
        Q = self.compute_Q_ip(small_img)
        return Q
    
    def filter_Q(self, Q):
        truncate = 3.5
        sigma = 1
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
    
    def sample_hessian(self, img, sampler):
        img = torch.from_numpy(img).unsqueeze(0).to(device).float()
        sampler = torch.from_numpy(sampler).unsqueeze(0).to(device).float()
        func = lambda x: self.master_func(img, x)
        handle_out = torch.autograd.functional.hvp(func, img, sampler)[1]
        del img, sampler
        torch.cuda.empty_cache()
        handle_out = handle_out.cpu().numpy().squeeze()
        return handle_out

    def prenormalize_img(self, img):
        img = data_normalize(img)
        img = img[0, :, :, :].numpy()
        return img
    
class compute_Q_layer(compute_Q_machines):

    def __init__(self, true_N, n_cwd, N=8):
        super().__init__(true_N, n_cwd, N)        
        self.sampling_depth = 128
        self.master_func_layer = lambda x, y, layer: resnet_loss_layer(x, y, layer)

    def compute_ood(self, img, layer, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x)

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
        hvp_faster = lambda x: self.sample_hessian(img, x)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, i]
            hvp = hvp_faster(rand_vec)
            return hvp

        Q = np.zeros((num[0], num[1], num[2]))
        for j in range(self.sampling_depth):
            Q += np.square(fun_eval(j))
        Q = np.sqrt(Q)
        Q_in = self.color_to_gray(Q)
        Q = Q_in.ravel('F')
        Q = self.normalize_Q(Q)
        Q = Q.reshape((num[1], num[2]), order='F')
        return Q

    def compute_Q_diag(self, img, layer, level=None):
        hvp_faster = lambda x: self.sample_hessian(img, x)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, i]
            hvp = hvp_faster(rand_vec)
            rand_vec = rand_vec.squeeze()
            hvp = hvp*rand_vec
            hvp_den = rand_vec*rand_vec
            return hvp, hvp_den
        
        Q = np.zeros((num[0], num[1], num[2]))
        normalizer = np.zeros((num[0], num[1], num[2]))
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
        img = self.prenormalize_img(img)
        self.master_func = lambda x, y: self.master_func_layer(x, y, layer)
        self.sampler = self.gauss_sample
        Q = self.compute_Q_ip(img, layer)
        return Q
    
    def sample_q_train_diag(self, img, layer, level=None):
        img = self.prenormalize_img(img)
        self.master_func = lambda x, y: self.master_func_layer(x, y, layer)
        self.sampler = self.gauss_sample
        Q = self.compute_Q_diag(img, layer)
        return Q

    def sample_hessian(self, img, sampler):
        img = torch.from_numpy(img).unsqueeze(0).to(device).float()
        sampler = torch.from_numpy(sampler).unsqueeze(0).to(device).float()
        func = lambda x: self.master_func(img, x)
        handle_out = torch.autograd.functional.hvp(func, img, sampler)[1]
        del img, sampler
        torch.cuda.empty_cache()
        handle_out = handle_out.cpu().numpy().squeeze()
        return handle_out
    
    def color_to_gray(self, Q):
        return np.sqrt(np.mean(Q**2, axis=0))
    
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
