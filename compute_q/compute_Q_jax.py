import numpy as np
import jax.numpy as jnp
import jax
from iqa_funs.jax_ssim import jax_ssim, jax_msssim, jax_ssim_zp
from jax.scipy.fft import dctn, idctn
from utils.q_utils_jax import brisque_func, compute_LPIPS_gs, niqe_func
from functools import partial


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


def reshape_image_blocks(image, block_size):
    # Get the shape of the original image
    height, width = image.shape

    # Calculate the number of blocks in both dimensions
    num_blocks_height = height // block_size
    num_blocks_width = width // block_size

    # Reshape the image into blocks
    reshaped_image = jnp.reshape(image[:num_blocks_height * block_size, :num_blocks_width * block_size],
                                  (num_blocks_height, block_size, num_blocks_width, block_size))

    # Transpose to have blocks in the first and third dimensions
    reshaped_image = jnp.transpose(reshaped_image, (0, 2, 1, 3))

    # Reshape to get the final result
    reshaped_image = jnp.reshape(reshaped_image, (num_blocks_height * num_blocks_width, block_size, block_size))

    return reshaped_image

def invert_reshape_image_blocks(reshaped_image, original_shape, block_size):
    # Get the shape of the original image
    num_blocks, _, _ = reshaped_image.shape

    # Calculate the number of blocks in both dimensions
    num_blocks_height, num_blocks_width = original_shape[0] // block_size, original_shape[1] // block_size

    # Reshape back to 4D array
    reshaped_image = reshaped_image.reshape((num_blocks_height, num_blocks_width, block_size, block_size))

    # Transpose to have blocks in the first and third dimensions
    reshaped_image = jnp.transpose(reshaped_image, (0, 2, 1, 3))

    # Reshape back to the original image
    original_image = reshaped_image.reshape((num_blocks_height * block_size, num_blocks_width * block_size))

    return original_image

@jax.jit
def inv_transform(img, mtx_trans):
    img = img.squeeze()
    N = np.sqrt(mtx_trans.shape[0]).astype(int)
    img_trans = jnp.zeros_like(img)
    mtx_trans = jnp.array(mtx_trans)

    def multiply_by_mtx(img_block):
        img_block = img_block.ravel('F')
        return (mtx_trans @ img_block).reshape((N, N), order='F')
    
    matrix_fun = jax.jit(jax.vmap(multiply_by_mtx))
    img_shape = img.shape
    # reshape img into blocks of N x N
    img = reshape_image_blocks(img, N)
    # apply the transformation to each block
    img_trans = matrix_fun(img)
    img_trans = invert_reshape_image_blocks(img_trans, img_shape, N)
    # undo the reshape

    """
    for i in range(img_trans.shape[0]//N):
        for j in range(img_trans.shape[1]//N):
            # multiply image block by the transformation matrix
            img_block = img[i*N:(i+1)*N, j*N:(j+1)*N]
            img_trans_tmp = multiply_by_mtx(img_block)
            img_trans = img_trans.at[i*N:(i+1)*N, j*N:(j+1)*N].set(img_trans_tmp)
    """
    img_trans = jnp.reshape(img_trans, (1, 1, img_trans.shape[0], img_trans.shape[1]))
    return img_trans


def fwd_dct(img, dct_size):
    N = dct_size
    img_trans = jnp.zeros_like(img)

    @jax.jit
    def compute_dct(img_block):
        return dctn(img_block, norm='ortho')

    for i in range(img_trans.shape[0]//N):
        for j in range(img_trans.shape[1]//N):
            # multiply image block by the transformation matrix
            img_block = compute_dct(img[i*N:(i+1)*N, j*N:(j+1)*N])
            img_trans = img_trans.at[i*N:(i+1)*N, j*N:(j+1)*N].set(img_block)
    return img_trans


#only jax the first argument
def inv_dct(img, dct_size):
    img = img.squeeze()
    N = dct_size
    img_trans = jnp.zeros_like(img)

    @jax.jit
    def compute_idct(img_block):
        return idctn(img_block, norm='ortho')

    for i in range(img_trans.shape[0]//N):
        for j in range(img_trans.shape[1]//N):
            # multiply image block by the transformation matrix
            img_block = compute_idct(img[i*N:(i+1)*N, j*N:(j+1)*N])
            img_trans = img_trans.at[i*N:(i+1)*N, j*N:(j+1)*N].set(img_block)
    img_trans = jnp.reshape(img_trans, (1, 1, img_trans.shape[0], img_trans.shape[1]))
    return img_trans

def to_tensor(img):
    img = jnp.array(img).reshape((1, 1, img.shape[0], img.shape[1]))
    return img


def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


def sample_hessian(func, img, sampler):
    fun_samp = lambda x: func(img, x)
    hessian = hvp(fun_samp, (img,), (sampler,))
    return hessian


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

        """
        Q = np.zeros(num)
        for j in range(self.sampling_depth):
            Q += np.square(np.array(fun_eval(j)))
        """
        #sampling_depth = 4
        vmap_fun_eval = jax.vmap(jax.jit(fun_eval))
        Q = jnp.sqrt(jnp.sum(vmap_fun_eval(jnp.arange(self.sampling_depth)), axis=0))
        #Q = jnp.sum(vmap_fun_eval(jnp.arange(self.sampling_depth)), axis=0)
        #Q = jnp.sqrt(Q)
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


    #@partial(jax.jit, static_argnums=(0,))
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

        var_num = 0
        var_den = 0
        mean_num = 0
        mean_den = 0
        run_idx = 0
        for j in range(self.sampling_depth):
            run_idx += 1
            sample_num = fun_eval_num(j)
            sample_den = fun_eval_den(j)
            mean_num += (sample_num - mean_num)/run_idx
            mean_den += (sample_den - mean_den)/run_idx
            if run_idx > 1:
                var_num += 1/(run_idx - 1)*((sample_num - mean_num)*(sample_num - mean_num) - var_num)
                var_den += 1/(run_idx - 1)*((sample_den - mean_den)*(sample_den - mean_den) - var_den)
        ood = var_num
        ood_den = var_den
        output = jnp.minimum(ood/ood_den, 1)
        return output

    def color_to_gray(self, img):
        return img
    
    def sample_ood_transformation(self, img, mtx_trans):
        img = self.prenormalize_img(img)
        img_trans = fwd_trans(img, mtx_trans)
        img_trans = to_tensor(img_trans)
        self.master_func = lambda x, y: self.aux_handle(inv_transform(x, mtx_trans), inv_transform(y, mtx_trans))
        ood = self.compute_ood(img_trans)
        return ood

    def sample_ood_dct(self, img, dct_size):
        img = to_tensor(img)
        img = self.prenormalize_img(img)
        img = jnp.array(img)
        img_trans = fwd_dct(img, dct_size)
        self.master_func = lambda x, y: self.aux_handle(inv_dct(x, dct_size), inv_dct(y, dct_size))
        ood = self.compute_ood(img_trans)

    def sample_ood(self, img, level=None):
        img = to_tensor(img)
        small_img = self.prenormalize_img(img)
        ood = self.compute_ood(small_img)
        # convert ood to a number
        ood = np.array(ood)
        return ood

    def just_sample_Q(self, img, level=None):
        img = self.prenormalize_img(img)
        self.sampler = self.rad

        img = to_tensor(img)
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)

        ind = np.random.randint(0, self.sampling_depth)
        rand_vec = self.sampler[:, :, :, :, ind]
        hvp = hvp_faster(rand_vec)
        hvp = np.array(hvp.reshape(self.true_N[0], self.true_N[1]))
        return hvp
    
    def sample_q(self, img, level=None):
        Q = self.sample_q_ip_radamacher(img)
        return Q

    def get_bunch_hvp(self, img):
        img = self.prenormalize_img(img)
        img = to_tensor(img)
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
        bunch = np.zeros((self.sampling_depth, self.true_N[0]*self.true_N[1]))
        for i in range(self.sampling_depth):
            sampler = self.rad[:, :, :, :, i]
            hvp = hvp_faster(sampler)
            bunch[i, :] = np.array(hvp).ravel('F')
        return bunch
    

    def get_sgd_hvp(self, img, iterations=1000, eta=0.05):
        img = self.prenormalize_img(img)  
        img = to_tensor(img)
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
        key = jax.random.PRNGKey(0)
        w = jax.random.normal(key, shape=(self.true_N[0]*self.true_N[1],))
        w = w / jnp.linalg.norm(w)
        for i in range(iterations):
            # plit the key
            print('iteration: ', i)
            key, subkey = jax.random.split(key)
            sampler = jax.random.normal(subkey, shape=(1, 1, self.true_N[0], self.true_N[1]))
            hvp = hvp_faster(sampler)
            hvp = hvp.ravel('F')
            inn_prod = jnp.dot(w, hvp)
            w = w + eta*inn_prod*hvp
            w = w / jnp.linalg.norm(w)
        w = np.array(w)
        return hvp
    
    # TODO(gilmer) This function should use higher numerical precision?
    def lanczos_alg(self, img, dim, order, rng_key):
        img = self.prenormalize_img(img)
        img = to_tensor(img)
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
        
    
        tridiag = jnp.zeros((order, order))
        vecs = jnp.zeros((order, dim))

        init_vec = jax.random.normal(rng_key, shape=(dim,))
        init_vec = init_vec / np.linalg.norm(init_vec)
        vecs = vecs.at[0].set(init_vec)

        beta = 0
        # TODO(gilmer): Better to use lax.fori loop for faster compile?
        for i in range(order):
            v = vecs[i, :].reshape((dim))
            if i == 0:
                v_old = 0
            else:
                v_old = vecs[i - 1, :].reshape((dim))

            v_call = v.reshape(img.shape)
            w = hvp_faster(v_call).squeeze().flatten()
            assert (w.shape[0] == dim and len(w.shape) == 1), (
                'Output of matrix_vector_product(v) must be of shape [dim].')
            w = w - beta * v_old

            alpha = jnp.dot(w, v)
            tridiag = tridiag.at[i, i].set(alpha)
            w = w - alpha * v

            # Full Reorthogonalization
            for j in range(i):
                tau = vecs[j, :].reshape((dim))
                coeff = jnp.dot(w, tau)
                w += -coeff * tau

            beta = jnp.linalg.norm(w)

            # TODO(gilmer): The tf implementation raises an exception if beta < 1e-6
            # here. However JAX cannot compile a function that has an if statement
            # that depends on a dynamic variable. Should we still handle this base?
            # beta being small indicates that the lanczos vectors are linearly
            # dependent.

            if i + 1 < order:
                tridiag = tridiag.at[i, i+1].set(beta)
                tridiag = tridiag.at[i+1, i].set(beta)
                vecs = vecs.at[i+1].set(w/beta)
        return (tridiag, vecs)

    def get_main_eigenvector(self, img):
        img = self.prenormalize_img(img)
        img = to_tensor(img)
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
        niters = 128
        sampler = self.rad[:, :, :, :, 0]
        sampler = sampler / np.linalg.norm(sampler)
        for i in range(niters):
            hvp = hvp_faster(sampler)
            sampler = hvp / np.linalg.norm(hvp)
        hvp = np.array(sampler.reshape(self.true_N[0], self.true_N[1]))
        return hvp
    
    def get_second_eigenvector(self, img):
        img = self.prenormalize_img(img)
        img = to_tensor(img)
        hvp_faster = lambda x: sample_hessian(self.master_func, img, x)
        niters = 128
        sampler = self.gauss_sample[:, :, :, :, 0]
        sampler = sampler / np.linalg.norm(sampler)
        for i in range(niters):
            hvp = hvp_faster(sampler)
            sampler = hvp / jnp.linalg.norm(hvp)
        eigval = jnp.sum(sampler*hvp_faster(sampler))
        main_eig = sampler
        #main_eig = jnp.array(sampler.reshape(self.true_N[0], self.true_N[1]))

        sampler = self.gauss_sample[:, :, :, :, 1]
        sampler = sampler / jnp.linalg.norm(sampler)
        for i in range(niters):
            hvp = hvp_faster(sampler)
            hvp = hvp - eigval*jnp.sum(sampler * main_eig) * main_eig
            sampler = hvp / jnp.linalg.norm(hvp)
        hvp = np.array(sampler.reshape(self.true_N[0], self.true_N[1]))
        main_eig = np.array(main_eig.reshape(self.true_N[0], self.true_N[1]))
        print('orthogonal: ', np.sum(hvp*main_eig), np.sum(hvp**2), np.sum(main_eig**2))
        eigval_2 = jnp.sum(sampler*hvp_faster(sampler))
        print('eigvals: ', eigval, eigval_2)
        return hvp



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
