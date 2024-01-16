import numpy as np
from utils.utils_lpit import dct_2d, idct_2d
from utils.utils_lpit import matlab_round
from my_codecs.SIC_class import SIC
from utils.bits_class import compute_bits_means

class JPEG(SIC):

    def __init__(self, nqs=12, N=8, center=True, uniform=False):
        # initialize the super class
        super().__init__(nqs, N, center, uniform)
        self.proy_Q_table()        

    def proy_Q_table(self):
        self.Q = []
        for j in range(self.nqs):
            self.Q.append(90*self.base_Q*self.quant[j])

    def set_Q(self, input):
        input = input.astype(np.float64)
        self.ind_closest = np.zeros((input.shape[0]//self.N, input.shape[1]//self.N))
        self.overhead_bits = 0

    def quant_layer(self, blk, Q, ind=None):
        quant_blk = matlab_round(128*blk/Q)
        return quant_blk
    
    def dequant_layer(self, quant_blk, Q, ind=None):
        dequant_blk = quant_blk*Q/128
        return dequant_blk

    def fwd_transform(self, input, ind=None, ind_qual=None):
        return dct_2d(input)
    
    def inv_transform(self, input, ind=None, ind_qual=None):
        return idct_2d(input)
