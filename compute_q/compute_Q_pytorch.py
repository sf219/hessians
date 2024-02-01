import torch
import numpy as np
from utils.q_utils import ssim_func, ms_ssim_func, haarpsi_func, brisque_func, compute_LPIPS_gs, resnet_loss
from utils.q_utils import resnet_loss_layer, vgg_loss_layer, alexnet_loss_layer
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fwd_trans(img, mtx_trans):
    N = np.sqrt(mtx_trans.shape[0]).astype(int)
    img_trans = np.zeros_like(img)
    for i in range(img_trans.shape[0]//N):
        for j in range(img_trans.shape[1]//N):
            # multiply image block by the transformation matrix
            img_block = img[i*N:(i+1)*N, j*N:(j+1)*N]
            img_block = img_block.ravel('F')
            img_trans_tmp = mtx_trans.T @ img_block
            img_trans[i*N:(i+1)*N, j*N:(j+1)*N] = img_trans_tmp.reshape((N, N), order='F')
    return img_trans

def inv_transform(img, mtx_trans):
    N = np.sqrt(mtx_trans.shape[0]).astype(int)
    img_trans = torch.zeros_like(img).to(device).type(torch.float32)
    mtx_trans = torch.from_numpy(mtx_trans).to(device).type(torch.float32)
    for i in range(img_trans.shape[2]//N):
        for j in range(img_trans.shape[3]//N):
            # multiply image block by the transformation matrix
            img_block = img[:, :, i*N:(i+1)*N, j*N:(j+1)*N]
            img_block = img_block.squeeze()
            img_block = img_block.t().ravel()
            img_trans_tmp = mtx_trans @ img_block
            img_trans_tmp = img_trans_tmp.reshape((N, N))
            img_trans[:, :, i*N:(i+1)*N, j*N:(j+1)*N] = img_trans_tmp.t().unsqueeze(0).unsqueeze(0)
    return img_trans


def to_tensor(img):
    if (len(img.shape) == 2):
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device).type(torch.float32)
    else:
        img = torch.from_numpy(img).unsqueeze(0).to(device).type(torch.float32)
    return img


def sample_hessian(master_func, img, sampler):
    img = to_tensor(img)
    sampler = to_tensor(sampler)
    func = lambda x: master_func(img, x)
    handle_out = torch.autograd.functional.hvp(func, img, sampler)[1]
    del img, sampler
    torch.cuda.empty_cache()
    out_val = handle_out.cpu().detach().numpy().squeeze()
    return out_val


class compute_Q():

    def __init__(self, true_N, channels=1, sampling_depth=128):
        self.true_N = true_N
        self.sampling_depth = sampling_depth
        self.channels = channels
        self.set_samplers()
        self.master_func = lambda x, y: torch.mean(torch.square(x - y))
        self.aux_handle = lambda x, y: torch.mean(torch.square(x - y))

    def set_samplers(self):
        # values needed to approximate the Hessian
        self.gauss_sample = np.random.randn(self.channels, self.true_N[0], self.true_N[1],
                                             self.sampling_depth)
        self.gauss_sample = np.array(self.gauss_sample)
        self.rad = np.sign(np.random.randn(self.channels, self.true_N[0], self.true_N[1], self.sampling_depth))

    def prenormalize_img(self, img):
        img = 2/255*img
        return img

    def compute_Q(self, img, level=None):
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)

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
        return Q
    
    
    def compute_Q_ip(self, img, level=None):
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, i].squeeze()
            hvp = hvp_faster(rand_vec)
            return hvp

        Q = np.zeros(num)
        for j in range(self.sampling_depth):
            Q += np.square(fun_eval(j))
        Q = np.sqrt(Q)
        Q = self.color_to_gray(Q)
        return Q

    def sample_q_ip_gaussian(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.gauss_sample
        Q = self.compute_Q_ip(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        return Q
  
    def sample_q_ip_radamacher(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.rad
        Q = self.compute_Q_ip(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        return Q
    
    def just_sample_Q(self, img, level=None):
        img = self.prenormalize_img(img)
        self.sampler = self.gauss_sample

        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
        num = img.shape

        ind = np.random.randint(0, self.sampling_depth)
        rand_vec = self.sampler[:, :, :, ind].squeeze()
        hvp = hvp_faster(rand_vec)
        return hvp

    def compute_Q_diag(self, img, level=None):
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, i].squeeze()
            hvp = hvp_faster(rand_vec)
            out = (hvp*rand_vec)
            return out

        Q = np.zeros(num)
        normalizer = np.zeros(num)
        for j in range(self.sampling_depth):
            Q += fun_eval(j)
            rand_vec = self.sampler[:, :, :, j].squeeze()
            normalizer += np.square(rand_vec)
        Q = Q / normalizer
        Q = self.color_to_gray(Q)
        return Q

    def color_to_gray(self, Q):
        return Q

    def sample_q_diag_gaussian(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.gauss_sample
        Q = self.compute_Q_diag(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        return Q
    
    def sample_q_diag_radamacher(self, img, level=None):
        small_img = self.prenormalize_img(img)
        self.sampler = self.rad
        Q = self.compute_Q_diag(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        return Q


    def compute_Q_det(self, img, level=None):
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
        num = img.shape

        def fun_eval(i, j):
            rand_vec = np.zeros(num)
            rand_vec[i, j] = 1
            hvp = hvp_faster(rand_vec)
            out = (hvp*rand_vec)
            return out

        Q = np.zeros(num)
        for j in range(num[0]):
            for i in range(num[1]):
                Q += fun_eval(j, i)
        Q = self.normalize_Q(Q)
        return Q
    
    def compute_ood(self, img, level=None):
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)

        def fun_eval(i):
            rand_vec = self.rad[:, :, :, i].squeeze()
            hvp = hvp_faster(rand_vec)
            hvp = np.sum(hvp * rand_vec)

            rand_vec = self.gauss_sample[:, :, :, i].squeeze() 
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
    

    def compute_half_ood(self, img, level=None):
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)

        def fun_eval(i):
            rand_vec = self.rad[:, :, :, i].squeeze()
            # set the upper left part to zero
            rand_vec[0:rand_vec.shape[0]//2, 0:rand_vec.shape[1]//2] = 0
            hvp = hvp_faster(rand_vec)
            hvp = np.sum(hvp * rand_vec)

            rand_vec = self.gauss_sample[:, :, :, i].squeeze() 
            rand_vec[0:rand_vec.shape[0]//2, 0:rand_vec.shape[1]//2] = 0
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
        Q = self.sample_q_diag_radamacher(img)
        return Q

    def sample_ood(self, img, level=None):
        small_img = self.prenormalize_img(img)
        ood = self.compute_ood(small_img)
        return ood

    def sample_half_ood(self, img, level=None):
        small_img = self.prenormalize_img(img)
        ood = self.compute_half_ood(small_img)
        return ood

    def sample_q_det(self, img, level=None):
        small_img = self.prenormalize_img(img)
        Q = self.compute_Q_det(small_img)
        return Q

    def check_first_derivative(self, img):
        img = self.prenormalize_img(img)
        img = to_tensor(img)
        img.requires_grad = True
        func = lambda x: self.master_func(img, x)
        img2 = torch.ones_like(img)
        handle_out = torch.autograd.functional.jvp(func, img, img2)[1]
        del img, img2
        torch.cuda.empty_cache()
        out_val = handle_out.cpu().numpy().squeeze()
        print(str(out_val))

    def check_evaluation(self, img):
        img = self.prenormalize_img(img)
        img = to_tensor(img)
        func = lambda x: self.master_func(img, x)
        handle_out = func(img)
        del img
        torch.cuda.empty_cache()
        out_val = handle_out.cpu().detach().numpy().squeeze()
        print(str(out_val))

    
    def compute_exact_hessian_cols(self, img, level=None):
            hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
            N = img.shape[0]
            samples = self.sampling_depth

            def fun_eval(i):
                rand_vec = np.zeros((N*N))
                rand_vec[i] = 0.01
                rand_vec = rand_vec.reshape(N, N)
                hvp = hvp_faster(rand_vec)*100
                hvp = hvp.ravel()
                return hvp
            
            H = np.zeros((N*N, samples))
            for i in range(samples):
                H[:, i] = fun_eval(i)
            return H
    
    def sample_ood_transformation(self, img, mtx_trans):
        img = self.prenormalize_img(img)
        img_trans = fwd_trans(img, mtx_trans)
        self.master_func = lambda x, y: self.aux_handle(inv_transform(x, mtx_trans), inv_transform(y, mtx_trans))
        ood = self.compute_ood(img_trans)
        return ood

class compute_Q_lpips(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)        
        self.master_func = lambda x, y: compute_LPIPS_gs(x, y)
        self.aux_handle = lambda x, y: compute_LPIPS_gs(x, y)

    def prenormalize_img(self, img):
        return super().prenormalize_img(img)-1
    

class compute_Q_ssim(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)
        self.master_func = lambda x, y: ssim_func(x, y)
        self.aux_handle = lambda x, y: ssim_func(x, y)


class compute_Q_msssim(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)
        self.master_func = lambda x, y: ms_ssim_func(x, y)
        self.aux_handle = lambda x, y: ms_ssim_func(x, y)



class compute_Q_haarpsi(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)
        self.master_func = lambda x, y: haarpsi_func(x, y)
        self.aux_handle = lambda x, y: haarpsi_func(x, y)


class compute_Q_brisque(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)
        self.master_func = lambda x, y: brisque_func(x, y)
        self.aux_handle = lambda x, y: brisque_func(x, y)

    def sample_q(self, img, level=None):
        Q = self.sample_q_ip_radamacher(img)
        return Q
    

class compute_Q_clip(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    def __init__(self, true_N):
        super().__init__(true_N)
    

data_normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class compute_Q_resnet18(compute_Q):

    def __init__(self, true_N, channels=3):
        super().__init__(true_N, channels)
        self.master_func = lambda x, y: resnet_loss(x, y)
        self.aux_handle = lambda x, y: resnet_loss(x, y)

    def to_tensor(self, img):
        return torch.from_numpy(img).unsqueeze(0).to(device).float()
    
    def prenormalize_img(self, img):
        img = data_normalize(img)
        img = img[0, :, :, :].numpy()
        return img

    def color_to_gray(self, Q):
        #weights = np.array([65.481, 128.553, 24.966])/255
        #Q = weights[0]*Q[0, :, :] + weights[1]*Q[1, :, :] + weights[2]*Q[2, :, :]
        Q = np.mean(Q, axis=0)
        return Q.squeeze()
        

class compute_Q_layer(compute_Q_resnet18):

    def __init__(self, true_N, channels=3):
        super().__init__(true_N, channels)
        self.master_func_layer = lambda x, y, layer: resnet_loss_layer(x, y, layer)
        self.aux_handle_layer = lambda x, y, layer: resnet_loss_layer(x, y, layer)

    def compute_ood(self, img, layer):
        self.master_func = lambda x, y: self.master_func_layer(x, y, layer)
        return super().compute_ood(img)
    
    def sample_q_ip_gaussian(self, img, layer, level=None):
        self.master_func = lambda x, y: self.master_func_layer(x, y, layer)
        return super().sample_q_ip_gaussian(img, level)
    
    def sample_q_diag_gaussian(self, img, layer, level=None):
        self.master_func = lambda x, y: self.master_func_layer(x, y, layer)
        return super().sample_q_diag_gaussian(img, level)
    
    def sample_q_diag_radamacher(self, img, layer, level=None):
        self.master_func = lambda x, y: self.master_func_layer(x, y, layer)
        return super().sample_q_diag_radamacher(img, level)
    
    def sample_q_ip_radamacher(self, img, layer, level=None):
        self.master_func = lambda x, y: self.master_func_layer(x, y, layer)
        return super().sample_q_ip_radamacher(img, level)


class compute_Q_layer_vgg16(compute_Q_layer):

    def __init__(self, true_N):
        super().__init__(true_N)        
        self.master_func_layer = lambda x, y, layer: vgg_loss_layer(x, y, layer)
        self.aux_handle_layer = lambda x, y, layer: vgg_loss_layer(x, y, layer)

class compute_Q_layer_alex(compute_Q_layer):

    def __init__(self, true_N):
        super().__init__(true_N)        
        self.master_func_layer = lambda x, y, layer: alexnet_loss_layer(x, y, layer)
        self.aux_handle_layer = lambda x, y, layer: alexnet_loss_layer(x, y, layer)