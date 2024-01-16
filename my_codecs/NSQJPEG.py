import numpy as np
from utils.utils_lpit import unifgrid
from utils.q_utils import decompose_Q
from utils.utils_lpit import inv_zig_zag
from scipy.fftpack import dct
from utils.utils_lpit import matlab_round
from scipy.linalg import eigh
from my_codecs.SIC_class import SIC
from utils.bits_class import Bits, compute_bits
from utils.coding_library import dpcm, rlgr, jdcenc


def eigen_decomp(Q, L):
    N = int(np.sqrt(L.shape[0]))
    Q1, Q2 = decompose_Q(Q)
    Q = np.kron(Q1, Q2)
    eigvals, eigvecs = eigh(L, Q, eigvals_only=False)
    eigvals = np.real(eigvals)
    # sort eig_vals in descending order
    eigvecs = np.real(eigvecs)
    D = dct(np.eye(N), norm='ortho', axis=0).T
    match = np.arange(N**2)
    N = int(np.sqrt(L.shape[0]))
    #Q1, Q2 = decompose_Q(Q)
    #Q = np.kron(Q1, Q2)
    eigvals, eigvecs = eigh(L, Q, eigvals_only=False)
    eigvals = np.real(eigvals)
    # sort eig_vals in descending order
    eigvecs = np.real(eigvecs)
    D = dct(np.eye(N), norm='ortho', axis=0).T
    help_mid_basis = np.zeros_like(eigvecs)
    for p in range(0, N**2):
        bas = np.zeros((N**2, N**2))
        bas[p, p] = 1
        bas = np.diag(bas)
        bas = np.reshape(bas, (N, N), order='F')
        bas = inv_zig_zag(bas)
        eig_ssp = D @ bas @ D.T
        eig_ssp = eig_ssp.ravel('F')
        help_mid_basis[:, p] = eig_ssp
    
    inner_prod = eigvecs.T @ Q @ help_mid_basis  
    match = np.ones((N**2))*(N**2 + 1)
    for p in range(0, N**2):
        vector = np.abs(inner_prod[:, p])
        pos = np.argsort(vector)[::-1]
        pos_max = 0
        match_tmp = pos[pos_max] 
        while match_tmp in match:
            pos_max += 1
            match_tmp = pos[pos_max]
        match[p] = match_tmp
    eigvecs = eigvecs[:, match.astype(np.int32)]

    D = np.kron(D, D)
    proy = eigvecs.T @ D
    sign_mtx = np.diag(np.sign(np.diag(proy)))
    eigvecs = eigvecs @ sign_mtx
    return eigvecs, eigvals


class NSQJPEG(SIC):

    def __init__(self, compute_Q_obj, nqs=12, N=8, center=True, uniform=True):
        super().__init__(nqs, N, center, uniform)
        self.compute_Q_obj = compute_Q_obj
        self.get_transform_basis()
        self.proy_Q_table()

    def compute_bits(self, trans, input=None, Q_list=None):
        tot_bits, bits_ac, bits_dc_prev = compute_bits(trans, self.N) 
        bits_over = tot_bits + self.overhead_bits - bits_dc_prev
        bits_dc = self.compute_bits_means(input, Q_list)
        bits_obj = Bits(bits_over+bits_dc, bits_ac, bits_dc)
        return bits_obj

    def proy_Q_table(self):
        self.Q = []
        D = dct(np.eye(self.N), norm='ortho', axis=0).T
        D = np.kron(D, D)
        for j in range(self.nqs):
            qf = self.quant[j]
            Q_inner = []
            for i in range(self.compute_Q_obj.n_cwd):
                tmp_Q = (self.base_Q)
                U = self.eigvecs_list[j][i]
                Uq = np.abs(D.T @ U)
                Uq = Uq @ np.linalg.inv(np.diag(np.sum(Uq, axis=0)))
                produ = np.abs(tmp_Q.ravel('F').T @ Uq)
                produ = produ.reshape((self.N, self.N), order='F')
                table = qf*90*produ
                Q_inner.append(table)
            self.Q.append(Q_inner)

    def get_transform_basis(self):
        L, _ = unifgrid(self.N)
        self.eigvecs_list = []
        self.eigvals_list = []
        self.q_mtx = []
        for j in range(self.nqs):
            centroids = self.compute_Q_obj.get_centroids(self.quant[j])
            inner_q = []
            inner = []
            inner_vals = []
            for i in range(self.compute_Q_obj.n_cwd):
                q_val = (centroids[i, :])
                inner_q.append(np.diag(q_val.ravel('F')))
                eigvecs, eigvals = eigen_decomp(np.diag(q_val.ravel('F')), L)
                #q_val = np.diag(np.kron(Q1, Q2)).reshape((self.N, self.N), order='F')
                inner.append(eigvecs)
                inner_vals.append(eigvals)
            self.eigvecs_list.append(inner)
            self.q_mtx.append(inner_q)
            self.eigvals_list.append(inner_vals)


    def set_Q(self, input, one_depth=True):
        input = input.astype(np.float64)
        if (one_depth is True):
            if (len(input.shape) == 3):
                input = input[:, :, 0]
        self.Qmtx, ind_closest, _ = self.compute_Q_obj.sample_q(input)
        self.overhead_bits = self.compute_Q_obj.compress_Q(ind_closest)
        print('overhead bits: ', self.overhead_bits)
        self.centroids = self.compute_Q_obj.get_centroids()
        self.ind_closest = ind_closest

    def quant_layer(self, blk, Q, ind=None):
        Q_norm = Q[ind]
        quant_blk = matlab_round(128*blk/Q_norm)
        return quant_blk
    
    def dequant_layer(self, quant_blk, Q, ind=None):
        Q_norm = Q[ind]
        dequant_blk = quant_blk*Q_norm/128
        return dequant_blk

    def fwd_transform(self, blk, ind, ind_qual=None):
        Q = self.q_mtx[ind_qual][ind]
        U = self.eigvecs_list[ind_qual][ind]
        trans_blk = U.T @ Q @ blk.ravel('F')
        return trans_blk.reshape(blk.shape, order='F')
    
    def inv_transform(self, blk, ind, ind_qual=None):
        U = self.eigvecs_list[ind_qual][ind]
        bck_trans = U @ blk.ravel('F')
        return bck_trans.reshape(blk.shape, order='F')

    def dpcm_smart(self, tmp):
        fdpcm = np.zeros_like(tmp)
        for j in range(tmp.shape[0]):
            tmp_vec = tmp[j, :]
            tmp_vec = tmp_vec.reshape((len(tmp_vec), 1))
            fdpcm[j, :] = dpcm(tmp_vec, 1)[0].squeeze()
        top = fdpcm[:, 0]
        top = top.reshape((len(top), 1))
        fdpcm[:, 0] = dpcm(top, 1)[0].squeeze()
        return fdpcm

    def compute_bits_means(self, input, Q_list):
        coefs_means = np.zeros((input.shape[0]//self.N, input.shape[1]//self.N, self.compute_Q_obj.n_cwd))
        for i in range(0, input.shape[0], self.N):
            for j in range(0, input.shape[1], self.N):
                for k in range(self.compute_Q_obj.n_cwd):
                    blk = input[i:i+self.N, j:j+self.N]
                    Q_mtx = self.q_mtx[0][k]
                    U_vec = self.eigvecs_list[0][k]
                    tmp = U_vec[:, 0].T @ Q_mtx @ blk.ravel('F') 
                    coefs_means[i//self.N, j//self.N, k] = matlab_round(128*tmp/(Q_list[k][0, 0]))
        coefs_means_in = coefs_means
        for k in range(self.compute_Q_obj.n_cwd):
            coefs_means[:, :, k] = self.dpcm_smart(coefs_means_in[:, :, k].squeeze())
        bits_dc = 0

        for k in range(self.compute_Q_obj.n_cwd):
            pos = self.ind_closest == k
            tier = coefs_means[:, :, k].squeeze()
            tocomp = tier[pos].ravel('F')
            #if tocomp.size == 0:
            #    continue
            #for p in range(len(tocomp)):
            #    bits_dc += len(jdcenc(np.array([tocomp[p].astype(int)])))
            tocomp = tocomp.astype(np.int32)
            bits_dc += len(rlgr(tocomp, 8)*8)
        return bits_dc
    

    def zig_zag_func(self, blk, ind=None):
        return blk
    
    def inv_zig_zag_func(self, blk, ind):
        return blk
        


