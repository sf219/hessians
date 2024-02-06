import numpy as np  
import scipy
from utils.utils_lpit import get_quantization_scales
from utils.utils_lpit import matlab_round, get_quantization_table, reshape_image_blocks, invert_reshape_image_blocks
from utils.bits_class import Bits, compute_bits
import jax
import jax.numpy as jnp
from utils.q_utils_jax import apply_zig_zag, inv_zig_zag
from functools import partial


class SIC:

    def __init__(self, nqs=12, N=8, center=True, uniform=False):
        self.quant, _ = get_quantization_scales(nqs, sup_val=90, inf_val=60)
        self.N = N
        self.nqs = nqs
        self.base_Q, self.base_C = get_quantization_table(N)
        self.mean_blks = 0
        if (uniform is True):
            self.base_Q = np.median(self.base_Q)*np.ones((self.N, self.N))/4
            self.base_C = np.median(self.base_C)*np.ones((self.N, self.N))/4
        self.center = center
        self.uniform = uniform

    def compute_bits(self, trans, input=None, Q_list=None):
        tot_bits, bits_ac, bits_dc = compute_bits(trans, self.N) 
        return tot_bits
    
    def fwd_transform(self, input, ind=None, ind_qual=None):
        pass

    def inv_transform(self, input, ind=None, ind_qual=None):
        pass

    def quant_layer(self, blk, Q, ind=None):
        pass

    def dequant_layer(self, quant_blk, Q, ind=None):
        pass

    def zig_zag_func(self, blk, inx=None):
        return apply_zig_zag(blk)
    
    def inv_zig_zag_func(self, blk, inx=None):
        return inv_zig_zag(blk)

    def compress_method(self, input, Q_list, flag_half=False):

        @jax.jit
        def quant_layer(quant_blk: jnp.ndarray, ind: jnp.int32) -> jnp.ndarray:
            return self.quant_layer(quant_blk, Q_list, ind)
        
        @jax.jit
        def trans_layer(input: jnp.ndarray, ind: jnp.int32) -> jnp.ndarray:
            return self.fwd_transform(input, ind)
        
        @jax.jit
        def zig_zag_layer(quant_blk: jnp.ndarray, ind: jnp.int32) -> jnp.ndarray:
            return self.zig_zag_func(quant_blk, ind)

        vec_quant_layer = jax.vmap(quant_layer, in_axes=(0, 0))
        vec_trans_layer = jax.vmap(trans_layer, in_axes=(0, 0))
        vec_zig_zag_layer = jax.vmap(zig_zag_layer, in_axes=(0, 0))

        input = input.astype(jnp.float32)
        if (self.center is True):
            input = 2*input/255 - 1
        N = self.N
        or_shape = input.shape
        img_blocks = reshape_image_blocks(input, N)

        if (flag_half is True):
            ind_vec = self.ind_closest_420.ravel()
        else:
            ind_vec = self.ind_closest.ravel()

        ind_vec = jnp.array(ind_vec).reshape((ind_vec.size,)).astype(jnp.int32)

        out_trans = vec_trans_layer(img_blocks, ind_vec)
        out_quant = vec_quant_layer(out_trans, ind_vec)
        out_zig_zag = vec_zig_zag_layer(out_quant, ind_vec)

        trans = invert_reshape_image_blocks(out_zig_zag, or_shape, N)
        trans_out = np.array(trans).astype(np.int32)
        bits = self.compute_bits(trans_out, input, Q_list)
        return trans_out, bits

    def uncompress_method(self, trans, Q_list, flag_half=False):
        #convert trans to float

        @jax.jit
        def dequant_layer(quant_blk: jnp.ndarray, ind: jnp.int32) -> jnp.ndarray:
            return self.dequant_layer(quant_blk, Q_list, ind)
        
        @jax.jit
        def inv_zig_zag_layer(quant_blk: jnp.ndarray, ind: jnp.int32) -> jnp.ndarray:
            return self.inv_zig_zag_func(quant_blk, ind)
        
        @jax.jit
        def inv_trans_layer(input: jnp.ndarray, ind: jnp.int32) -> jnp.ndarray:
            return self.inv_transform(input, ind)

        vec_dequant_layer = jax.vmap(dequant_layer, in_axes=(0, 0))
        vec_inv_zig_zag_layer = jax.vmap(inv_zig_zag_layer, in_axes=(0, 0))
        vec_inv_trans_layer = jax.vmap(inv_trans_layer, in_axes=(0, 0))

        trans = trans.astype(jnp.float32)
        N = self.N
        or_shape = trans.shape
        trans_blks = reshape_image_blocks(trans, N)
        output = np.zeros_like(trans_blks)
        if (flag_half is True):
            ind_vec = self.ind_closest_420.ravel()
        else:
            ind_vec = self.ind_closest.ravel()

        ind_vec = ind_vec.reshape((ind_vec.size,)).astype(jnp.int32)
        out_inv_zig_zag = vec_inv_zig_zag_layer(trans_blks, ind_vec)
        out_dequant = vec_dequant_layer(out_inv_zig_zag, ind_vec)
        output = vec_inv_trans_layer(out_dequant, ind_vec)

        output = invert_reshape_image_blocks(output, or_shape, N)
        output = np.array(output)
        if (self.center is True):
            output = 255*(output + 1)/2
            output = np.clip(matlab_round(output), 0, 255)
        return output
    

    ## here we have different methods to handle the input

    def compress_420(self, img, ind_qual):
        self.ind_qual = ind_qual
        Q_list = self.Q[ind_qual]
        trans_y, bits_1 = self.compress_method(jnp.array(img[:, :, 0]), Q_list)

        Q_list = self.chroma_Q[ind_qual]
        cb = img[:, :, 1]
        cr = img[:, :, 2]
        cb_compress = np.zeros((cb.shape[0]//2, cb.shape[1]//2))
        cr_compress = np.zeros((cr.shape[0]//2, cr.shape[1]//2))
        # take blocks of size 2x2 and average the result
        for i in range(0, cb.shape[0], 2):
            for j in range(0, cb.shape[1], 2):
                cb_compress[i//2, j//2] = np.mean(cb[i:i+2, j:j+2])
                cr_compress[i//2, j//2] = np.mean(cr[i:i+2, j:j+2])
        trans_cb, bits_2 = self.compress_method(jnp.array(cb_compress), Q_list, flag_half=True)
        trans_cr, bits_3 = self.compress_method(jnp.array(cr_compress), Q_list, flag_half=True)
        bits = bits_1 + bits_2 + bits_3 + self.overhead_bits
        # stack components together
        output = (trans_y, trans_cb, trans_cr)
        bits_arr = (bits_1 + self.overhead_bits, bits_2, bits_3)
        return output, bits, bits_arr

    def uncompress_420(self, trans, ind_qual):
        Q_list = self.Q[ind_qual]
        out_y = self.uncompress_method(jnp.array(trans[0]),Q_list)
        Q_list = self.chroma_Q[ind_qual]

        cb_compress = self.uncompress_method(jnp.array(trans[1]), Q_list, flag_half=True)
        cr_compress = self.uncompress_method(jnp.array(trans[2]), Q_list, flag_half=True)
        cb = np.zeros((cb_compress.shape[0]*2, cb_compress.shape[1]*2))
        cr = np.zeros((cr_compress.shape[0]*2, cr_compress.shape[1]*2))
        # take blocks of size 2x2 and copy the results
        cb = scipy.ndimage.zoom(cb_compress, 2, order=1)
        cr = scipy.ndimage.zoom(cr_compress, 2, order=1)

        output = np.zeros((out_y.shape[0], out_y.shape[1], 3))
        output[:, :, 0] = out_y
        output[:, :, 1] = cb
        output[:, :, 2] = cr
        return output
    
    def compress_400(self, img, ind_qual):
        self.ind_qual = ind_qual
        Q_list = self.Q[ind_qual]
        trans, bits = self.compress_method(img, Q_list)
        bits = bits + self.overhead_bits
        return trans, bits

    def uncompress_400(self, trans, ind_qual):
        Q_list = self.Q[ind_qual]
        output = self.uncompress_method(trans, Q_list)
        return output
    
    def compress_444(self, img, ind_qual):
        self.ind_qual = ind_qual
        output = np.zeros_like(img)
        Q_list = self.Q[ind_qual]
        output[:, :, 0], bits_1 = self.compress_method(img[:, :, 0], Q_list)
        Q_list = self.chroma_Q[ind_qual]
        output[:, :, 1], bits_2 = self.compress_method(img[:, :, 1], Q_list)
        output[:, :, 2], bits_3 = self.compress_method(img[:, :, 2], Q_list)
        bits = bits_1 + bits_2 + bits_3 + self.overhead_bits
        return output, bits
    
    def uncompress_444(self, trans, ind_qual):
        output = np.zeros_like(trans)
        Q_list = self.Q[ind_qual]
        output[:, :, 0] = self.uncompress_method(trans[:, :, 0], Q_list)
        Q_list = self.chroma_Q[ind_qual]
        output[:, :, 1] = self.uncompress_method(trans[:, :, 1], Q_list)
        output[:, :, 2] = self.uncompress_method(trans[:, :, 2], Q_list)
        return output
