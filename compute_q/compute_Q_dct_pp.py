import torch
import torch_dct as dct
from utils.utils_lpit import dct_2d
from compute_q.compute_Q_pytorch import compute_Q
from compute_q.compute_Q_machines import compute_Q_layer_mnist
from utils.q_utils import ssim_func, ms_ssim_func, haarpsi_func, brisque_func, compute_LPIPS_gs, clip_func, mnist_loss_layer


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class compute_Q_dct(compute_Q):
    
    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.dct = lambda x: dct.dct_2d(x,  norm='ortho')
        self.idct = lambda x: dct.idct_2d(x,  norm='ortho')

    def sample_hessian(self, img1, img2):
        img1 = dct_2d(img1)
        img1 = torch.from_numpy(img1).unsqueeze(0).unsqueeze(0).to(device).type(torch.float32)
        img2 = torch.from_numpy(img2).unsqueeze(0).unsqueeze(0).to(device).type(torch.float32)

        func = lambda x: self.master_func(img1, x)
        handle_out = torch.autograd.functional.hvp(func, img1, img2)[1]
        del img1, img2
        torch.cuda.empty_cache()
        out_val = handle_out.cpu().detach().numpy().squeeze()
        return out_val


class compute_Q_ssim_dct(compute_Q_dct):

    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: ssim_func(self.idct(x), self.idct(y))


class compute_Q_msssim_dct(compute_Q_dct):

    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: ms_ssim_func(self.idct(x), self.idct(y))


class compute_Q_lpips_dct(compute_Q_dct):

    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: compute_LPIPS_gs(self.idct(x), self.idct(y))

    
class compute_Q_brisque_dct(compute_Q_dct):

    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: brisque_func(self.idct(x), self.idct(y))

class compute_Q_layer_mnist_dct(compute_Q_layer_mnist):

    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.sampling_depth = 256
        self.dct = lambda x: dct.dct_2d(x,  norm='ortho')
        self.idct = lambda x: dct.idct_2d(x,  norm='ortho')
        self.master_func = lambda x, y, layer: mnist_loss_layer(self.dct(x), self.dct(y), layer)
        self.set_samplers()

    def sample_hessian(self, img, sampler, layer):
        img = dct_2d(img)
        img = torch.from_numpy(img).unsqueeze(0).to(device).float()
        sampler = torch.from_numpy(sampler).unsqueeze(0).to(device).float()
        func = lambda x: self.master_func(img, x, layer)
        handle_out = torch.autograd.functional.hvp(func, img, sampler)[1]
        del img, sampler
        torch.cuda.empty_cache()
        handle_out = handle_out.cpu().numpy().squeeze()
        return handle_out
