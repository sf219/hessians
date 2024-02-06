import numpy as np
from utils.utils_lpit import unifgrid
from utils.utils_lpit import inv_zig_zag
from scipy.fftpack import dct
from utils.utils_lpit import matlab_round
from scipy.linalg import eigh
from my_codecs.SIC_class import SIC
from utils.bits_class import compute_bits, dpcm_smart
from utils.coding_library import jacenc, jdcenc

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

    def __init__(self, compute_Q_obj, q_ops_obj, nqs=12, N=8, center=True, uniform=True, quant_scal=90):
        super().__init__(nqs, N, center, uniform)
        self.compute_Q_obj = compute_Q_obj
        self.q_ops_obj = q_ops_obj
        self.quant_scal = quant_scal

    def compute_bits(self, trans, input=None, Q_list=None):
        tot_bits, _, bits_dc_prev = compute_bits(trans, self.N) 
        bits_dc = self.compute_bits_means(input, Q_list)
        tot_bits = tot_bits - bits_dc_prev + bits_dc
        return tot_bits

    def proy_Q_table(self):
        self.Q = []
        self.chroma_Q = []
        for j in range(self.nqs):
            qf = self.quant[j]
            Q_inner = []
            chroma_Q_inner = []
            for i in range(self.q_ops_obj.n_cwd):
                U = self.eigvecs_list[i]
                produ_Q = proy_Q_table(self.base_Q, U)
                produ_C = proy_Q_table(self.base_C, U)
                
                """
                if self.uniform is True:
                    produ_Q = np.ones((N, N))
                    produ_C = np.ones((N, N))
                """
                table_Q = qf*self.quant_scal*produ_Q
                table_C = qf*self.quant_scal*produ_C

                Q_inner.append(table_Q)
                chroma_Q_inner.append(table_C)
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

    def set_Q(self, input, one_depth=True, lbr_mode=False):
        if (one_depth is True):
            if (len(input.shape) == 3):
                input = input[:, :, 0]
        img_quan = 2*input/255
        self.Qmtx = self.compute_Q_obj.sample_q(input)
        self.Qmtx = self.q_ops_obj.normalize_q(self.Qmtx)
        self.q_ops_obj.quantize_q(self.Qmtx, img_quan)
        self.q_ops_obj.choose_ncwd(lbr_mode)
        self.overhead_bits = self.q_ops_obj.overhead_bits
        print(' overhead bits: ', self.overhead_bits)
        self.centroids = self.q_ops_obj.centroids
        self.ind_closest = self.q_ops_obj.ind_closest
        self.ind_closest_420 = self.q_ops_obj.ind_closest_420
        self.Q_quantized = self.q_ops_obj.Q
        self.set_basis()

    def quant_layer(self, blk, Q, ind=None):
        Q_norm = Q[ind]
        quant_blk = matlab_round(128*blk/Q_norm)
        return quant_blk
    
    def dequant_layer(self, quant_blk, Q, ind=None):
        Q_norm = Q[ind]
        dequant_blk = quant_blk*Q_norm/128
        return dequant_blk

    def fwd_transform(self, blk, ind):
        Q = self.q_mtx[ind]
        U = self.eigvecs_list[ind]
        trans_blk = U.T @ Q @ blk.ravel('F')
        return trans_blk.reshape(blk.shape, order='F')
    
    def inv_transform(self, blk, ind):
        U = self.eigvecs_list[ind]
        bck_trans = U @ blk.ravel('F')
        return bck_trans.reshape(blk.shape, order='F')


    def compute_bits_means(self, input, Q_list):
        coefs_means = np.zeros((input.shape[0]//self.N, input.shape[1]//self.N, self.q_ops_obj.n_cwd))
        for i in range(0, input.shape[0], self.N):
            for j in range(0, input.shape[1], self.N):
                for k in range(self.q_ops_obj.n_cwd):
                    blk = input[i:i+self.N, j:j+self.N]
                    Q_mtx = self.q_mtx[k]
                    U_vec = self.eigvecs_list[k]
                    tmp = U_vec[:, 0].T @ Q_mtx @ blk.ravel('F') 
                    coefs_means[i//self.N, j//self.N, k] = matlab_round(128*tmp/(Q_list[k][0, 0]))
        coefs_means_dpcm = np.zeros_like(coefs_means)

        avg_coefs = np.zeros((self.q_ops_obj.n_cwd))
        for k in range(self.q_ops_obj.n_cwd):
            avg_coefs[k] = np.mean(np.diag(self.q_mtx[k]))*128/Q_list[k][0, 0]
        
        avg_coefs = avg_coefs/np.sum(avg_coefs)

        for k in range(self.q_ops_obj.n_cwd):
            coefs_means_dpcm[:, :, k] = dpcm_smart(coefs_means[:, :, k].squeeze())

        # the only possible scenario when they don't match is when using 4:2:0 chroma subsampling
        if (self.ind_closest.shape[0] == coefs_means.shape[0]):
            ind_closest = self.ind_closest
            flag_color = False
        else:
            ind_closest = self.ind_closest_420
            flag_color = True

        
        final_coefs = np.zeros((coefs_means.shape[0], coefs_means.shape[1]))
        for i in range(coefs_means.shape[0]):
            for j in range(coefs_means.shape[1]):
                pos = ind_closest[i, j].astype(int)
                final_coefs[i, j] = coefs_means_dpcm[i, j, pos]
        

        bits_dc = len(jacenc(final_coefs.ravel('F').astype(int)))

        """
        for k in range(self.q_ops_obj.n_cwd):
            pos = ind_closest == k
            tier = coefs_means_dpcm[:, :, k].squeeze()
            tocomp = tier[pos].ravel('F')
            if tocomp.size == 0:
                continue
            # extend the array to be a multiple of 16
            if avg_coefs[k] < 0.5 or flag_color is True:
                tocomp = tocomp.astype(int)
                bits_dc += len(jacenc(tocomp))
            else:
                for i in range(0, tocomp.size):
                    bits_dc += len(jdcenc(np.array([tocomp[i]])))
        """
        return bits_dc
    

    def zig_zag_func(self, blk, ind=None):
        return blk
    
    def inv_zig_zag_func(self, blk, ind):
        return blk
        


