import numpy as np
from utils.utils_lpit import unifgrid
from utils.utils_lpit import inv_zig_zag
from scipy.fftpack import dct
from utils.q_utils_jax import matlab_round
from scipy.linalg import eigh
from my_codecs.SIC_jax import SIC
from utils.bits_class import compute_bits, dpcm_smart
from utils.coding_library import dpcm, rlgr, jdcenc, jacenc
from functools import partial
import jax
import jax.numpy as jnp

# we are working with JPEG, so it's safe to set N = 8
N = 8
D_1d = dct(np.eye(N), norm='ortho', axis=0).T
D_2d = np.kron(D_1d, D_1d)


def fix_sign(basis):
    proy = basis.T @ D_2d
    sign_mtx = np.diag(np.sign(np.diag(proy)))
    basis = basis @ sign_mtx
    return basis


def gevd(L, Q):
    eigvals, eigvecs = eigh(L, Q, eigvals_only=False)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    return eigvals, eigvecs


def sort_inv_zz_dct_basis():
    help_mid_basis = np.zeros((N*N, N*N))
    for p in range(0, N**2):
        bas = np.zeros((N**2, N**2))
        bas[p, p] = 1
        bas = np.diag(bas)
        bas = np.reshape(bas, (N, N), order='F')
        bas = inv_zig_zag(bas)
        eig_ssp = D_1d @ bas @ D_1d.T
        eig_ssp = eig_ssp.ravel('F')
        help_mid_basis[:, p] = eig_ssp
    return help_mid_basis


def find_matches(inner_prod):
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
    return match


def compute_iagft_basis(Q, L):
    eigvals, eigvecs = gevd(L, Q)
    help_mid_basis = sort_inv_zz_dct_basis()
    inner_prod = eigvecs.T @ Q @ help_mid_basis  
    match = find_matches(inner_prod)
    eigvecs = eigvecs[:, match.astype(np.int32)]
    eigvecs = fix_sign(eigvecs)
    return eigvecs, eigvals


def proy_Q_table(table, basis):
    Uq = np.abs(D_2d.T @ basis)
    Uq = Uq @ np.linalg.inv(np.diag(np.sum(Uq, axis=0)))
    produ = np.abs(table.ravel('F').T @ Uq)
    produ = produ.reshape((N, N), order='F')
    return produ




class NSQJPEG(SIC):

    def __init__(self, compute_Q_obj, q_ops_obj, nqs=12, N=8, center=True, uniform=True):
        super().__init__(nqs, N, center, uniform)
        self.compute_Q_obj = compute_Q_obj
        self.q_ops_obj = q_ops_obj

    def compute_bits(self, trans, input=None, Q_list=None):
        tot_bits, _, bits_dc_prev = compute_bits(trans, self.N) 
        bits_dc = self.compute_bits_means(input, Q_list)
        tot_bits = tot_bits - bits_dc_prev + bits_dc
        return tot_bits

    def proy_Q_table(self):
        self.Q = []
        self.chroma_Q = []
        D = dct(np.eye(self.N), norm='ortho', axis=0).T
        D = np.kron(D, D)
        for j in range(self.nqs):
            qf = self.quant[j]
            Q_inner = []
            chroma_Q_inner = []
            for i in range(self.q_ops_obj.n_cwd):
                U = self.eigvecs_list[i]

                produ = proy_Q_table(self.base_Q, U)
                table = qf*90*produ
                Q_inner.append(table)

                produ = proy_Q_table(self.base_C, U)
                table = qf*90*produ
                chroma_Q_inner.append(table)
            self.Q.append(Q_inner)
            self.chroma_Q.append(chroma_Q_inner)

    def set_basis(self):
        self.get_transform_basis()
        self.proy_Q_table()

    def get_transform_basis(self):
        L, _ = unifgrid(self.N)
        self.eigvecs_list = []
        self.eigvals_list = []
        self.q_mtx = []
        centroids = self.q_ops_obj.get_centroids()
        for i in range(self.q_ops_obj.n_cwd):
            q_val = (centroids[i, :])
            eigvecs, eigvals = compute_iagft_basis(np.diag(q_val.ravel('F')), L)
            self.eigvecs_list.append(eigvecs)
            self.q_mtx.append(np.diag(q_val.ravel('F')))
            self.eigvals_list.append(eigvals)

    def set_Q(self, input, one_depth=True):
        if (one_depth is True):
            if (len(input.shape) == 3):
                input = input[:, :, 0]
        self.Qmtx = self.compute_Q_obj.sample_q(input)
        self.Qmtx = self.q_ops_obj.normalize_q(self.Qmtx)
        self.q_ops_obj.quantize_q(self.Qmtx, input)
        self.q_ops_obj.choose_ncwd()
        self.overhead_bits = self.q_ops_obj.overhead_bits
        print(' overhead bits: ', self.overhead_bits)
        self.centroids = self.q_ops_obj.centroids
        self.ind_closest = self.q_ops_obj.ind_closest
        self.ind_closest_420 = self.q_ops_obj.ind_closest_420
        self.Q_quantized = self.q_ops_obj.Q
        self.set_basis()

    def set_Q_qual_level(self, qual_level):
        self.q_ops_obj.choose_ncwd_qlevel(qual_level)
        self.overhead_bits = self.q_ops_obj.overhead_bits
        print('level:', qual_level, ' overhead bits: ', self.overhead_bits)
        self.centroids = self.q_ops_obj.centroids
        self.ind_closest = self.q_ops_obj.ind_closest
        self.ind_closest_420 = self.q_ops_obj.ind_closest_420
        self.Q = self.q_ops_obj.Q

    @partial(jax.jit, static_argnums=(0,))
    def quant_layer(self, blk, Q, ind=None):
        Q_norm = jnp.array(Q)[ind]
        quant_blk = matlab_round(128*blk/Q_norm)
        return quant_blk
    
    @partial(jax.jit, static_argnums=(0,))
    def dequant_layer(self, quant_blk, Q, ind=None):
        Q_norm = jnp.array(Q)[ind]
        dequant_blk = quant_blk*Q_norm/128
        return dequant_blk

    @partial(jax.jit, static_argnums=(0,))
    def fwd_transform(self, blk, ind):
        Q = jnp.array(self.q_mtx)[ind]
        U = jnp.array(self.eigvecs_list)[ind]
        trans_blk = U.T @ Q @ blk.ravel('F')
        return trans_blk.reshape(blk.shape, order='F')
    
    @partial(jax.jit, static_argnums=(0,))
    def inv_transform(self, blk, ind):
        U = jnp.array(self.eigvecs_list)[ind]
        bck_trans = U @ blk.ravel('F')
        return bck_trans.reshape(blk.shape, order='F')


    def compute_bits_means(self, input, Q_list):
        coefs_means = np.zeros((input.shape[0]//self.N, input.shape[1]//self.N, self.q_ops_obj.n_cwd))

        for k in range(self.q_ops_obj.n_cwd):
            Q_mtx = self.q_mtx[k]
            U_vec = self.eigvecs_list[k]

            def compute_dc(blk):
                tmp = U_vec[:, 0].T @ Q_mtx @ blk.ravel('F') 
                return matlab_round(128*tmp/(Q_list[k][0, 0]))
            
            for i in range(0, input.shape[0], self.N):
                for j in range(0, input.shape[1], self.N):
                    blk = input[i:i+self.N, j:j+self.N]
                    coefs_means[i//self.N, j//self.N, k] = compute_dc(blk)

        coefs_means_in = coefs_means
        for k in range(self.q_ops_obj.n_cwd):
            coefs_means[:, :, k] = dpcm_smart(coefs_means_in[:, :, k].squeeze())
        bits_dc = 0

        # the only possible scenario when they don't match is when using 4:2:0 chroma subsampling
        if (self.ind_closest.shape[0] == coefs_means.shape[0]):
            ind_closest = self.ind_closest
        else:
            ind_closest = self.ind_closest_420

        for k in range(self.q_ops_obj.n_cwd):
            pos = ind_closest == k
            tier = coefs_means[:, :, k].squeeze()
            tocomp = tier[pos].ravel('F')
            if tocomp.size == 0:
                continue
            tocomp = tocomp.astype(int)
            bits_dc += len(jacenc(tocomp))
        return bits_dc
    

    def zig_zag_func(self, blk, ind=None):
        return blk
    
    def inv_zig_zag_func(self, blk, ind):
        return blk
        


