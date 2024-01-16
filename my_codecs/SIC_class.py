import numpy as np  
from utils.utils_lpit import get_quantization_scales, apply_zig_zag, inv_zig_zag
from utils.utils_lpit import matlab_round, get_quantization_table
from utils.bits_class import Bits, compute_bits

class SIC:

    def __init__(self, nqs=12, N=8, center=True, uniform=False):
        self.quant, _ = get_quantization_scales(nqs, sup_val=90, inf_val=50)
        self.N = N
        self.nqs = nqs
        self.base_Q = get_quantization_table(N)
        self.mean_blks = 0
        if (uniform is True):
            self.base_Q = np.min(self.base_Q)*np.ones((self.N, self.N))
        self.center = center
        self.uniform = uniform

    def compute_bits(self, trans, input=None, Q_list=None):
        tot_bits, bits_ac, bits_dc = compute_bits(trans, self.N) 
        bits_over = tot_bits + self.overhead_bits
        bits_obj = Bits(bits_over, bits_ac, bits_dc)
        return bits_obj
    
    def fwd_transform(self, input, ind=None, ind_qual=None):
        pass

    def inv_transform(self, input, ind=None, ind_qual=None):
        pass

    def quant_layer(self, blk, Q, ind=None):
        pass

    def dequant_layer(self, quant_blk, Q, ind=None):
        pass

    def zig_zag_func(self, blk, inx=None):
        return apply_zig_zag(blk, self.N)
    
    def inv_zig_zag_func(self, blk, inx=None):
        return inv_zig_zag(blk, self.N)
    

    def compress_method(self, input, Q_list):
        input = input.astype(np.float64)
        if (self.center is True):
            input = 2*input/255 - 1
        trans = np.zeros_like(input)
        N = self.N
        for i in range(0, input.shape[0], N):
            for j in range(0, input.shape[1], N):
                blk = input[i:i+self.N, j:j+N]
                ind = int(self.ind_closest[i//N, j//N])
                trans_blk = self.fwd_transform(blk, ind, 0)
                quant_blk = self.quant_layer(trans_blk, Q_list, ind)
                zz_blk = self.zig_zag_func(quant_blk, ind)
                trans[i:i+N, j:j+N] = zz_blk
        trans_out = trans.astype(np.int32)
        bits = self.compute_bits(trans_out, input, Q_list)
        return trans_out, bits


    def uncompress_method(self, trans, Q_list):
        #convert trans to float
        trans = trans.astype(np.float64)
        output = np.zeros_like(trans)
        N = self.N
        for i in range(0, trans.shape[0], N):
            for j in range(0, trans.shape[1], N):
                ind = int(self.ind_closest[i//N, j//N])
                trans_blk = trans[i:i+N, j:j+N]
                izz_blk = self.inv_zig_zag_func(trans_blk, ind)
                dequant = self.dequant_layer(izz_blk, Q_list, ind)
                bck_trans = self.inv_transform(dequant, ind, 0)
                output[i:i+N, j:j+N] = bck_trans
        if (self.center is True):
            output = 255*(output + 1)/2
            output = np.clip(matlab_round(output), 0, 255)
        return output

    def compress(self, input, depth, ind_qual):
        trans = np.zeros_like(input)
        Q_list = self.Q[ind_qual]
        if (depth==1):
            trans, bits = self.compress_method(input, Q_list)
        else:
            bits = np.zeros(depth)
            for j in range(depth):
                if j == 0:
                    trans[:, :, j], bits_tmp = self.compress_method(input[:, :, j], Q_list)
                    bits[j] = bits_tmp.bits_over
                else:
                    trans[:, :, j], bits_tmp = self.compress_method(input[:, :, j], Q_list)
                    bits[j] = bits_tmp.bits_over
            bits = np.sum(bits)
        return trans, bits
    
    def uncompress(self, trans, depth, ind_qual):
        output = np.zeros_like(trans)
        Q_list = self.Q[ind_qual]
        if(depth==1):
            output = self.uncompress_method(trans, Q_list)
        else:
            for j in range(depth):
                output[:, :, j] = self.uncompress_method(trans[:, :, j], Q_list)
        return output