from iqa_funs.brisque_jax import brisque_loss
from iqa_funs.niqe_jax import niqe_loss
import jax.numpy as jnp
import jax
from lpips_jax_master.lpips_jax.lpips import LPIPSEvaluator
from functools import partial
from jax.scipy.stats import norm

@jax.jit
def brisque_func(img1, img2):
    #kernel_size = 7
    #pad = (kernel_size-1)//2
    #img1 = jnp.pad(img1.squeeze(), pad, mode='symmetric') 
    #img2 = jnp.pad(img2.squeeze(), pad, mode='symmetric')
    #img1 = img1.reshape((1, 1, img1.shape[0], img1.shape[1]))
    #img2 = img2.reshape((1, 1, img2.shape[0], img2.shape[1]))
    bris_1 = brisque_loss(img1)
    bris_2 = brisque_loss(img2)
    diff = (bris_2 - bris_1)
    cdf = norm.cdf(diff) - 0.1
    term_2 = diff**2 * cdf
    #term_2 = 1/2 * (diff + jnp.sqrt(jnp.square(diff) + 1e-4))
    #term_2 = (bris_2 - bris_1)**2
   # term_2 = torch.maxium(bris_2 - bris_1, torch.tensor(0))
    return term_2


@jax.jit
def niqe_func(img1, img2):
    #kernel_size = 7
    #pad = (kernel_size-1)//2
    #img1 = jnp.pad(img1.squeeze(), pad, mode='symmetric') 
    #img2 = jnp.pad(img2.squeeze(), pad, mode='symmetric')
    #img1 = img1.reshape((1, 1, img1.shape[0], img1.shape[1]))
    #img2 = img2.reshape((1, 1, img2.shape[0], img2.shape[1]))
    bris_1 = niqe_loss(img1)
    bris_2 = niqe_loss(img2)
    diff = -(bris_2 - bris_1)
    cdf = norm.cdf(diff) - 0.1
    term_2 = diff**2 * cdf
    #term_2 = 1/2 * (diff + jnp.sqrt(jnp.square(diff) + 1e-4))
    #term_2 = (bris_2 - bris_1)**2
   # term_2 = torch.maxium(bris_2 - bris_1, torch.tensor(0))
    return term_2


lp_jax = LPIPSEvaluator(replicate=False)

@jax.jit
def compute_LPIPS_gs(img1, img2):
    img1 = img1.squeeze()
    img2 = img2.squeeze()
    true_N = img1.shape
    img1 = jnp.repeat(img1[:, :, jnp.newaxis], 3, axis=2)
    img1 = img1.transpose(2, 0, 1)
    img1 = img1.reshape(1, 3, true_N[0], true_N[1])
    img1 = jnp.array(img1).transpose(0, 2, 3, 1)
    img2 = jnp.repeat(img2[:, :, jnp.newaxis], 3, axis=2)
    img2 = img2.transpose(2, 0, 1)
    img2 = img2.reshape(1, 3, true_N[0], true_N[1])
    img2 = jnp.array(img2).transpose(0, 2, 3, 1)
    lpips = lp_jax(img1, img2)
    lpips = lpips[0][0][0][0]
    return lpips


