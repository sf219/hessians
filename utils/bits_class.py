from utils.coding_library import dpcm, jacenc, jdcenc, rlgr, irlgr
import numpy as np
import matplotlib.pyplot as plt


class Bits:
    def __init__(self, bits_over=0, bits_ac=0, bits_dc=0):
        self.bits_over = bits_over
        self.bits_ac = bits_ac
        self.bits_dc = bits_dc
    
    def print_bits(self):
        print('Bits overhead: ', self.bits_over)
        print('Bits AC: ', self.bits_ac)
        print('Bits DC: ', self.bits_dc)


class IntegerTransform():
    def __init__(self):
        self.C = np.array([[1, 1, 1, 1],
                            [2, 1, -1, -2],
                            [1, -1, -1, 1],
                            [1, -2, 2, -1]], dtype=np.float32)

    def forward(self, X):
        W = np.matmul(np.matmul(self.C, X), self.C.T)
        return W


def compute_bits_means(means, N=4):
    int_trans = IntegerTransform()
    means_trans = np.zeros_like(means)
    for i in range(0, means_trans.shape[0], N):
        for j in range(0, means_trans.shape[1], N):
            means_trans[i:i+N, j:j+N] = int_trans.forward(means[i:i+N, j:j+N])
    dccof, _, _ = compute_bits(means_trans.astype(int), 4)
    return dccof


def compute_bits_no_means(trans, N=8):
    nb = trans.shape[0]//N
    mb = trans.shape[1]//N
    acseq = np.zeros((0), dtype=int)
    for i in range(mb):
        for j in range(nb):
            tmp = trans[i*N:(i+1)*N, j*N:(j+1)*N]
            vec_tmp = np.ravel(tmp, 'F')
            if (np.all(vec_tmp == 0)):
                acseq = np.concatenate((acseq, np.array([999]).astype(int)))
                continue
            eobi = np.where(vec_tmp != 0)[0][-1]
            tocat = np.append(vec_tmp[0:eobi+1], 999).astype(int)
            acseq = np.concatenate((acseq, tocat))
    accof = jacenc(acseq)
    bpps = (len(accof))/(trans.size)
    return bpps


def dpcm_smart(tmp):
    fdpcm = np.zeros_like(tmp, dtype=np.int32)
    for j in range(tmp.shape[0]):
        tmp_vec = tmp[j, :]
        tmp_vec = tmp_vec.reshape((len(tmp_vec), 1))
        tmp_vec = tmp_vec.astype(np.int32)
        fdpcm[j, :] = dpcm(tmp_vec, 1).squeeze()
    top = fdpcm[:, 0]
    top = top.reshape((len(top), 1))
    fdpcm[:, 0] = dpcm(top, 1).squeeze()
    return fdpcm

def compute_bits_omega(trans, N=8):
    nb = trans.shape[0]//N
    mb = trans.shape[1]//N
    tmp = trans[::N, ::N]
    dccof = np.zeros((0), dtype=int)
    eobis = []
    fdpcm = dpcm_smart(tmp)
    fdpcm_vec = fdpcm.ravel()
    for i in range(mb*nb):
        t2 = jdcenc(np.array([fdpcm_vec[i]]))
        dccof = np.concatenate((dccof, t2))
    breakpoint()    


def compute_bits(trans, N=8):
    nb = trans.shape[0]//N
    mb = trans.shape[1]//N
    tmp = trans[::N, ::N]
    dccof = np.zeros((0), dtype=int)
    eobis = []
    fdpcm = dpcm_smart(tmp)
    bits_dc = 0
    for i in range(nb):
        for j in range(mb):
            t2 = jdcenc(np.array([fdpcm[i, j]]))
            dccof = np.concatenate((dccof, t2))
            bits_dc += len(t2)
    bits_ac = 0
    count = 0
    no_zeros = 0
    for i in range(nb):
        for j in range(mb):
            tmp = trans[i*N:(i+1)*N, j*N:(j+1)*N]
            vec_tmp = np.ravel(tmp, 'F')
            if (np.all(vec_tmp == 0)):
                #dccof = np.concatenate((dccof, np.array([0]).astype(int)))
                #acseq = np.concatenate((acseq, np.array([999]).astype(int)))
                bits_ac += 4 # add 1010
                continue
            eobi = np.where(vec_tmp != 0)[0][-1]
            eobis.append(eobi)
            tmp_dp = vec_tmp[1:eobi+1]
            no_zeros += np.sum(tmp_dp!=0)
            tocat = np.append(tmp_dp, 999).astype(int)
            if len(tmp_dp)>0:
                if (tmp_dp[0] == 0):
                    count+=1
                #    print(tocat)
            seq = jacenc(tocat)
            bits_ac += len(seq)
    bits_ac = bits_ac
    bpps = bits_ac + bits_dc
    return bpps, bits_ac, bits_dc


def compute_bits_block(trans, N=8):
    nb = trans.shape[0]//N
    mb = trans.shape[1]//N
    tmp = trans[::N, ::N]
    dccof = np.zeros((0), dtype=int)
    eobis = []
    fdpcm = dpcm_smart(tmp)
    bits = np.zeros((nb, mb))
    for i in range(nb):
        for j in range(mb):
            t2 = jdcenc(np.array([fdpcm[i, j]]))
            dccof = np.concatenate((dccof, t2))
            #bits[i, j] += len(t2)
    bits_ac = 0
    count = 0
    no_zeros = 0
    for i in range(nb):
        for j in range(mb):
            tmp = trans[i*N:(i+1)*N, j*N:(j+1)*N]
            vec_tmp = np.ravel(tmp, 'F')
            if (np.all(vec_tmp == 0)):
                bits_ac += 4
                continue
            eobi = np.where(vec_tmp != 0)[0][-1]
            eobis.append(eobi)
            tmp_dp = vec_tmp[1:eobi+1]
            no_zeros += np.sum(tmp_dp!=0)
            tocat = np.append(tmp_dp, 999).astype(int)
            if len(tmp_dp)>0:
                if (tmp_dp[0] == 0):
                    count+=1
                #    print(tocat)
            seq = jacenc(tocat)
            bits[i, j] += len(seq)
    return bits


def subband_encoder(trans, N=8):
    tmp = trans[::N, ::N]
    bits_ac = 0
    fdpcm = np.zeros_like(tmp)
    for j in range(tmp.shape[0]):
        tmp_vec = tmp[j, :]
        tmp_vec = tmp_vec.reshape((len(tmp_vec), 1))
        fdpcm[j, :] = dpcm(tmp_vec, 1)[0].squeeze()
    top = fdpcm[:, 0]
    top = top.reshape((len(top), 1))
    fdpcm[:, 0] = dpcm(top, 1)[0].squeeze()
    fdpcm = fdpcm.ravel()
    bits_dc = 0
    for i in range(len(fdpcm)):
        t2 = jdcenc(np.array([fdpcm[i]]))
        bits_dc += len(t2)

    for i in range(N):
        for j in range(N):
            if (i == 0 and j == 0):
                continue
            vec = trans[i::N, j::N]
            vec = np.ravel(vec, 'F')
            accof = rlgr(vec.astype(np.int32), 3)
            bits_ac += len(accof)*8

    print('Bits AC: ', bits_ac/trans.size)
    return bits_ac+bits_dc, bits_ac, bits_dc

def compute_bits_beta(trans, N=8):
    nb = trans.shape[0]//N
    mb = trans.shape[1]//N
    tmp = trans[::N, ::N]
    bits_dc = compute_bits_means(tmp)
    acseq = np.zeros((0), dtype=int)
    eobis = []
    for i in range(mb):
        for j in range(nb):
            tmp = trans[i*N:(i+1)*N, j*N:(j+1)*N]
            vec_tmp = np.ravel(tmp, 'F')
            if (np.all(vec_tmp == 0)):
                acseq = np.concatenate((acseq, np.array([999]).astype(int)))
                continue
            eobi = np.where(vec_tmp != 0)[0][-1]
            eobis.append(eobi)
            tocat = np.append(vec_tmp[1:eobi+1], 999).astype(int)
            acseq = np.concatenate((acseq, tocat))
    accof = jacenc(acseq)
    mean_acseq = np.mean(acseq[acseq != 999])
    std_acseq = np.std(acseq[acseq != 999])
    len_acseq = len(acseq[acseq != 999])
    eobis = np.array(eobis)
    #print('Mean AC: ', mean_acseq, 'Std AC: ', std_acseq, 'Len AC: ', len_acseq, 'Mean EOB: ', np.mean(eobis))
    bits_ac = len(accof)/(trans.size)
    bpps = (len(accof) + bits_dc)/(trans.size)
    return bpps, bits_ac, bits_dc


def compute_bits_inds(trans, inds, n_cwd, N=8):
    nb = trans.shape[0]//N
    mb = trans.shape[1]//N
    tmp = trans[::N, ::N]
    fdc = np.ravel(tmp, 'C')
    fdc = np.reshape(fdc, (len(fdc), 1))
    inds_vec = np.ravel(inds, 'C')
    dccof = np.zeros((0), dtype=int)
    bits_ac = 0
    for k in range(n_cwd):
        acseq = np.zeros((0), dtype=int)
        fdc_class = fdc[inds_vec == k]
        fdpcm, _ = dpcm(fdc_class, 1)
        kk = 0
        for i in range(mb):
            for j in range(nb):
                if (inds[i, j] != k):
                    continue
                tmp = trans[i*N:(i+1)*N, j*N:(j+1)*N]
                vec_tmp = np.ravel(tmp, 'F')
                if (np.all(vec_tmp == 0)):
                    dccof = np.concatenate((dccof, np.array([0]).astype(int)))
                    acseq = np.concatenate((acseq, np.array([999]).astype(int)))
                    continue
                eobi = np.where(vec_tmp != 0)[0][-1]
                tocat = np.append(vec_tmp[1:eobi+1], 999).astype(int)
                t2 = jdcenc(fdpcm[kk])
                kk+=1
                acseq = np.concatenate((acseq, tocat))
                dccof = np.concatenate((dccof, t2))
        bits_ac += len(jacenc(acseq))

    bits_ac_out = bits_ac/(trans.size)
    bits_dc = len(dccof)/(trans.size)
    bpps = (bits_ac + len(dccof))/(trans.size)
    return bpps, bits_ac_out, bits_dc


def compute_bits_intra(trans, N=8):
    nb = trans.shape[0]//N
    mb = trans.shape[1]//N
    tmp = trans[::N, ::N]
    fdc = np.ravel(tmp, 'C')
    fdc = np.reshape(fdc, (len(fdc), 1))
    fdpcm, _ = dpcm(fdc, 1)
    acseq = np.zeros((0), dtype=int)
    dccof = np.zeros((0), dtype=int)
    for i in range(mb):
        for j in range(nb):
            tmp = trans[i*N:(i+1)*N, j*N:(j+1)*N]
            vec_tmp = np.ravel(tmp, 'F')
            if (np.all(vec_tmp == 0)):
                dccof = np.concatenate((dccof, np.array([0]).astype(int)))
                acseq = np.concatenate((acseq, np.array([999]).astype(int)))
                continue
            eobi = np.where(vec_tmp != 0)[0][-1]
            tocat = np.append(vec_tmp[1:eobi+1], 999).astype(int)
            t2 = jdcenc(fdpcm[i*nb+j])
            acseq = np.concatenate((acseq, tocat))
            dccof = np.concatenate((dccof, t2))
    accof = jacenc(acseq)
    eobi = np.where(fdpcm != 0)[0][-1]
    tocat = np.append(fdpcm[0:eobi], 999).astype(int)
    t2 = jacenc(tocat)
    bpps = (len(accof) + len(t2))/(trans.size)
    return bpps


def compute_bits_vec(trans, N=8):
    fdc = trans[:, 0]
    fdc = np.reshape(fdc, (len(fdc), 1))
    fdpcm, _ = dpcm(fdc, 1)
    acseq = np.zeros((0), dtype=int)
    dccof = np.zeros((0), dtype=int)
    for i in range(trans.shape[0]):
        vec_tmp = trans[i, :]
        if (np.all(vec_tmp == 0)):
            dccof = np.concatenate((dccof, np.array([0]).astype(int)))
            acseq = np.concatenate((acseq, np.array([999]).astype(int)))
            continue
        eobi = np.where(vec_tmp != 0)[0][-1]
        tocat = np.append(vec_tmp[1:eobi+1], 999).astype(int)
        t2 = jdcenc(fdpcm[i])
        acseq = np.concatenate((acseq, tocat))
        dccof = np.concatenate((dccof, t2))
    accof = jacenc(acseq)
    bpps = (len(accof) + len(dccof))/(trans.size)
    return bpps


def compute_bits_vec_out(trans, N=8):
    fdc = trans[:, 0]
    fdc = np.reshape(fdc, (len(fdc), 1))
    fdpcm, _ = dpcm(fdc, 1)
    dccof = rlgr(fdpcm)
    vecs = []
    bits_ac = 0

    for j in range(trans.shape[1]-1):
        fdc = trans[:, j+1]
        if (np.all(fdc == 0)):
            vecs.append(np.array([0]).astype(int))
            continue
        eobi = np.where(fdc != 0)[0][-1]
        tocat = np.append(fdc[1:eobi+1], 999).astype(int)
        vecs = np.concatenate((vecs, tocat))
    vecs = np.array(vecs)
    bits_ac += len(jacenc(vecs.astype(int)))
    bpps = (bits_ac + 8*len(dccof))/(trans.size)
    print('DC compression Q: ', len(dccof)*8, 'AC compression Q: ', bits_ac)
    return bpps
