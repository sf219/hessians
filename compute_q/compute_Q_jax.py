import numpy as np
import jax.numpy as jnp
import jax
from iqa_funs.jax_ssim import jax_ssim, jax_msssim, jax_ssim_zp
from utils.q_utils_jax import brisque_func, compute_LPIPS_gs, niqe_func, to_tensor, hvp, sample_hessian
from utils.q_utils_jax import block_multiply
from functools import partial


class compute_Q():

    def __init__(self, true_N, channels=1, sampling_depth=128):
        self.true_N = true_N
        self.sampling_depth = sampling_depth
        self.channels = channels
        self.set_samplers()
        self.master_func = lambda x, y: jnp.mean(jnp.square(x-y))
        self.aux_handle = lambda x, y: jnp.mean(jnp.square(x-y))

    def set_samplers(self):
        self.gauss_sample = np.random.randn(self.channels, self.true_N[0], self.true_N[1],
                                             self.sampling_depth)
        self.rad = np.sign(np.random.randn(self.channels, self.true_N[0], self.true_N[1], self.sampling_depth))

        self.gauss_sample = jnp.array(self.gauss_sample).reshape((1, 1, self.true_N[0], self.true_N[1], self.sampling_depth))
        self.rad = jnp.array(self.rad).reshape((1, 1, self.true_N[0], self.true_N[1], self.sampling_depth))
        self.sampler = (self.rad).astype(jnp.float32)

    def prenormalize_img(self, img):
        img = 2/255 * img
        return img

    def compute_Q_diag(self, img: jnp.ndarray):
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
        num = img.shape

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, :, i]
            hvp = hvp_faster(rand_vec)
            out = (hvp*rand_vec)
            return out        
        
        Q = jnp.zeros(num)
        normalizer = jnp.zeros(num)
        for j in range(self.sampling_depth):
            Q += fun_eval(j)
            rand_vec = self.sampler[:, :, :, :, j]
            normalizer += jnp.square(rand_vec)
        Q = Q / normalizer
        Q = self.color_to_gray(Q)
        return Q

    def sample_q_diag_radamacher(self, img):
        img = to_tensor(img)
        small_img = self.prenormalize_img(img)
        Q = self.compute_Q_diag(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        Q = np.array(Q)
        return Q

    @partial(jax.jit, static_argnums=(0,))
    def compute_Q_ip(self, img: jnp.ndarray):
        hvp_faster = lambda x: self.master_func(img, x)

        def fun_eval(i):
            rand_vec = self.sampler[:, :, :, :, i]
            hvp_mtx = jnp.square(hvp(hvp_faster, (img,), (rand_vec,)))
            return hvp_mtx

        vmap_fun_eval = jax.vmap(jax.jit(fun_eval))
        Q = jnp.sqrt(jnp.sum(vmap_fun_eval(jnp.arange(self.sampling_depth)), axis=0))
        Q = self.color_to_gray(Q)
        return Q

    def sample_q_ip_gaussian(self, img):
        img = to_tensor(img)
        small_img = self.prenormalize_img(img)
        self.sampler = self.gauss_sample
        Q = self.compute_Q_ip(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        Q = np.array(Q)
        return Q
  
    def sample_q_ip_radamacher(self, img, level=None):
        img = to_tensor(img)
        small_img = self.prenormalize_img(img)
        Q = self.compute_Q_ip(small_img)
        Q = Q.reshape(self.true_N[0], self.true_N[1])
        Q = np.array(Q)
        return Q


    @partial(jax.jit, static_argnums=(0,))
    def compute_ood(self, img: jnp.ndarray):
        hvp_faster = lambda x: self.master_func(img, x)

        def fun_eval_num(i):
            rand_vec = self.rad[:, :, :, :, i]
            hvp_mtx = hvp(hvp_faster, (img,), (rand_vec,)) * rand_vec
            hvp_mtx = jnp.sum(hvp_mtx)
            return hvp_mtx
        
        def fun_eval_den(i):
            rand_vec = self.gauss_sample[:, :, :, :, i]
            hvp_den = hvp(hvp_faster, (img,), (rand_vec,)) * rand_vec
            hvp_den = jnp.sum(hvp_den)
            return hvp_den

        fun_num = jax.vmap(fun_eval_num)
        fun_den = jax.vmap(fun_eval_den)

        samples_num = fun_num(jnp.arange(self.sampling_depth))
        samples_den = fun_den(jnp.arange(self.sampling_depth))

        ood_num = jnp.var(samples_num)
        ood_den = jnp.var(samples_den)
        output = jnp.minimum(ood_num/ood_den, 1)
        return output

    def color_to_gray(self, img):
        return img
    
    def sample_ood_transformation(self, img, mtx_trans):
        img = self.prenormalize_img(img)
        img = to_tensor(img)
        img_trans = block_multiply(img, mtx_trans.T)
        self.master_func = lambda x, y: self.aux_handle(block_multiply(x, mtx_trans), block_multiply(y, mtx_trans))
        ood = self.compute_ood(img_trans)
        return ood

    def sample_ood(self, img):
        img = to_tensor(img)
        small_img = self.prenormalize_img(img)
        ood = self.compute_ood(small_img)
        # convert ood to a number
        ood = np.array(ood)
        return ood

    def just_sample_Q(self, img):
        img = self.prenormalize_img(img)
        self.sampler = self.rad

        img = to_tensor(img)
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)

        ind = np.random.randint(0, self.sampling_depth)
        rand_vec = self.sampler[:, :, :, :, ind]
        hvp = hvp_faster(rand_vec)
        hvp = np.array(hvp.reshape(self.true_N[0], self.true_N[1]))
        return hvp
    
    def sample_q(self, img):
        Q = self.sample_q_ip_radamacher(img)
        return Q
    

class compute_Q_ssim(compute_Q):

    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)
        self.master_func = lambda x, y: jax_ssim(x, y)
        self.aux_handle = lambda x, y: jax_ssim(x, y)

class compute_Q_ssim_zp(compute_Q):

    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)
        self.master_func = lambda x, y: jax_ssim_zp(x, y)
        self.aux_handle = lambda x, y: jax_ssim_zp(x, y)


class compute_Q_msssim(compute_Q):

    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)
        self.master_func = lambda x, y: jax_msssim(x, y)
        self.aux_handle = lambda x, y: jax_msssim(x, y)


class compute_Q_brisque(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)
        self.master_func = lambda x, y: brisque_func(x, y)
        self.aux_handle = lambda x, y: brisque_func(x, y)
    

class compute_Q_niqe(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)
        self.master_func = lambda x, y: niqe_func(x, y)
        self.aux_handle = lambda x, y: niqe_func(x, y)


class compute_Q_lpips(compute_Q):
    # this class is a child of the compute_Q class
    # it inherits all the methods of compute_Q, but it changes the compute_Q method
    def __init__(self, true_N, channels=1, sampling_depth=128):
        super().__init__(true_N, channels, sampling_depth)   
        self.master_func = lambda x, y: compute_LPIPS_gs(x, y)
        self.aux_handle = lambda x, y: compute_LPIPS_gs(x, y)

    def prenormalize_img(self, img):
        return super().prenormalize_img(img)-1
