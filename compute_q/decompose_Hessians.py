from compute_q.compute_Q_jax import compute_Q_jax
import numpy as np
import jax
import jax.numpy as jnp
from utils.q_utils_jax import sample_hessian, to_tensor

class decompose_Hessians(compute_Q_jax):

    def __init__(self, true_N, sampling_depth):
        super().__init__(true_N, sampling_depth)

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
    