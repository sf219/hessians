
import numpy as np
from compute_q.compute_Q_jax import compute_Q_niqe, compute_Q_brisque, compute_Q_ssim, compute_Q_msssim, compute_Q_lpips
from utils.q_utils import ssim_func as ssim_mod
from utils.q_utils import ms_ssim_func as ms_ssim_mod
import jax
import jax.numpy as jnp

n_cwd = 4
true_N = (512, 768)
nqs = 6
N = 8

###### REMEMBER THAT NSQJPEG IS MODIFIED TO NOT INCLUDE RLGR ENCODING ######
###### 
flag_uniform = True

compute_Qs = [compute_Q_niqe, compute_Q_brisque, compute_Q_ssim, compute_Q_msssim, compute_Q_lpips]
for compute_Q_class in compute_Qs:
    compute_Q_obj = compute_Q_class(true_N, sampling_depth=64)
    #q_ops_obj = compute_ops_class(true_N=true_N, n_cwd=n_cwd, N=8, nqs=nqs)

    def hvp(f, primals, tangents):
        return jax.jvp(jax.grad(f), primals, tangents)[1]

    img = np.random.rand(1, 1, true_N[0], true_N[1])
    img = jnp.array(img)

    img2 = np.random.rand(1, 1, true_N[0], true_N[1], 16)
    img2 = jnp.array(img2)


    master_func = lambda x: compute_Q_obj.master_func(img, x)

    def run_hessians():
            def fun_eval(i):
                rand_vec = img2[:, :, :, :, i]
                hvp_mtx = jnp.square(hvp(master_func, (img,), (rand_vec,)))
                return hvp_mtx
            
            vmap_fun_eval = jax.vmap(fun_eval)
            Q = jnp.sqrt(jnp.sum(vmap_fun_eval(jnp.arange(16)), axis=0))
            return Q

    compiled = jax.jit(run_hessians).lower()
    flops_hessian = compiled.cost_analysis()['flops']

    master_func = lambda x: compute_Q_obj.master_func(img, x)
    compiled = jax.jit(master_func).lower(img)
    flops_metric = compiled.cost_analysis()['flops']
    print('Hessian: ', flops_hessian/(true_N[0]*true_N[1]), 'Metric: ', flops_metric/(true_N[0]*true_N[1]))