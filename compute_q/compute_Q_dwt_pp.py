import torch
import torch_dct as dct
import pywt
from pywt import Wavelet
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple, Union
from compute_q.compute_Q_pytorch import compute_Q
from utils.q_utils import ssim_func, ms_ssim_func, haarpsi_func, brisque_func, compute_LPIPS_gs, clip_func


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


wave = pywt.Wavelet('db4')

dec_hi = torch.Tensor(wave.dec_hi[::-1]) 
dec_lo = torch.Tensor(wave.dec_lo[::-1])
rec_hi = torch.Tensor(wave.rec_hi)
rec_lo = torch.Tensor(wave.rec_lo)

def _get_pad(data_len: int, filt_len: int) -> Tuple[int, int]:
    """Compute the required padding.

    Args:
        data_len (int): The length of the input vector.
        filt_len (int): The size of the used filter.

    Returns:
        Tuple: The first entry specifies how many numbers
            to attach on the right. The second
            entry covers the left side.

    """
    # pad to ensure we see all filter positions and
    # for pywt compatability.
    # convolution output length:
    # see https://arxiv.org/pdf/1603.07285.pdf section 2.3:
    # floor([data_len - filt_len]/2) + 1
    # should equal pywt output length
    # floor((data_len + filt_len - 1)/2)
    # => floor([data_len + total_pad - filt_len]/2) + 1
    #    = floor((data_len + filt_len - 1)/2)
    # (data_len + total_pad - filt_len) + 2 = data_len + filt_len - 1
    # total_pad = 2*filt_len - 3

    # we pad half of the total requried padding on each side.
    padr = (2 * filt_len - 3) // 2
    padl = (2 * filt_len - 3) // 2

    # pad to even singal length.
    if data_len % 2 != 0:
        padr += 1

    return padr, padl

def _get_len(wavelet: Union[Tuple[torch.Tensor, ...], str, Wavelet]) -> int:
    """Get number of filter coefficients for various wavelet data types."""
    if isinstance(wavelet, tuple):
        return wavelet[0].shape[0]
    else:
        return len(_as_wavelet(wavelet))
    
def _fwt_pad2(
    data: torch.Tensor, wavelet: Union[Wavelet, str], mode: str = "reflect"
) -> torch.Tensor:
    """Pad data for the 2d FWT.

    This function pads along the last two axes.

    Args:
        data (torch.Tensor): Input data with 4 dimensions.
        wavelet (Wavelet or str): A pywt wavelet compatible object or
            the name of a pywt wavelet.
        mode (str): The padding mode.
            Supported modes are::

                "reflect", "zero", "constant", "periodic", "symmetric".

            "reflect" is the default mode.

    Returns:
        The padded output tensor.

    """
    mode = _translate_boundary_strings(mode)
    wavelet = _as_wavelet(wavelet)
    padb, padt = _get_pad(data.shape[-2], _get_len(wavelet))
    padr, padl = _get_pad(data.shape[-1], _get_len(wavelet))
    if mode == "symmetric":
        data_pad = _pad_symmetric(data, [(padt, padb), (padl, padr)])
    else:
        data_pad = torch.nn.functional.pad(data, [padl, padr, padt, padb], mode=mode)
    return data_pad

def _pad_symmetric_1d(signal: torch.Tensor, pad_list: Tuple[int, int]) -> torch.Tensor:
    padl, padr = pad_list
    dimlen = signal.shape[0]
    if padl > dimlen or padr > dimlen:
        if padl > dimlen:
            signal = _pad_symmetric_1d(signal, (dimlen, 0))
            padl = padl - dimlen
        if padr > dimlen:
            signal = _pad_symmetric_1d(signal, (0, dimlen))
            padr = padr - dimlen
        return _pad_symmetric_1d(signal, (padl, padr))
    else:
        cat_list = [signal]
        if padl > 0:
            cat_list.insert(0, signal[:padl].flip(0))
        if padr > 0:
            cat_list.append(signal[-padr::].flip(0))
        return torch.cat(cat_list, axis=0)  # type: ignore

def _pad_symmetric(
    signal: torch.Tensor, pad_lists: List[Tuple[int, int]]
) -> torch.Tensor:
    if len(signal.shape) < len(pad_lists):
        raise ValueError("not enough dimensions to pad.")

    dims = len(signal.shape) - 1
    for pos, pad_list in enumerate(pad_lists[::-1]):
        current_axis = dims - pos
        signal = signal.transpose(0, current_axis)
        signal = _pad_symmetric_1d(signal, pad_list)
        signal = signal.transpose(current_axis, 0)
    return signal

def _translate_boundary_strings(pywt_mode: str) -> str:
    """Translate pywt mode strings to PyTorch mode strings.

    We support constant, zero, reflect, and periodic.
    Unfortunately, "constant" has different meanings in the
    Pytorch and PyWavelet communities.

    Raises:
        ValueError: If the padding mode is not supported.

    """
    if pywt_mode == "constant":
        pt_mode = "replicate"
    elif pywt_mode == "zero":
        pt_mode = "constant"
    elif pywt_mode == "reflect":
        pt_mode = pywt_mode
    elif pywt_mode == "periodic":
        pt_mode = "circular"
    elif pywt_mode == "symmetric":
        # pytorch does not support symmetric mode,
        # we have our own implementation.
        pt_mode = pywt_mode
    else:
        raise ValueError("Padding mode not supported.")
    return pt_mode


def _as_wavelet(wavelet: Union[Wavelet, str]) -> Wavelet:
    """Ensure the input argument to be a pywt wavelet compatible object.

    Args:
        wavelet (Wavelet or str): The input argument, which is either a
            pywt wavelet compatible object or a valid pywt wavelet name string.

    Returns:
        Wavelet: the input wavelet object or the pywt wavelet object described by the
            input str.
    """
    if isinstance(wavelet, str):
        return pywt.Wavelet(wavelet)
    else:
        return wavelet
    
def _outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Torch implementation of numpy's outer for 1d vectors."""
    a_flat = torch.reshape(a, [-1])
    b_flat = torch.reshape(b, [-1])
    a_mul = torch.unsqueeze(a_flat, dim=-1)
    b_mul = torch.unsqueeze(b_flat, dim=0)
    return a_mul * b_mul

filters = torch.stack([_outer(dec_lo, dec_lo),
                       _outer(dec_hi, dec_lo),
                       _outer(dec_lo, dec_hi),
                       _outer(dec_hi, dec_hi)], dim=0).to(device)

inv_filters = torch.stack([_outer(rec_lo, rec_lo),
                           _outer(rec_hi, rec_lo),
                           _outer(rec_lo, rec_hi),
                           _outer(rec_hi, rec_hi)], dim=0).to(device)


def wt(vimg, levels=1):
    padded = _fwt_pad2(vimg, wave)  
    #padded = torch.nn.functional.pad(vimg,(pad,pad,pad,pad), mode='reflect')
    res = torch.nn.functional.conv2d(padded, filters[:,None],stride=2)
    if levels>1:
        res[:,:1] = wt(res[:,:1],levels-1)
    h = res.size(2)*(2**levels)
    w = res.size(3)*(2**levels)
    res = res.view(-1,2,h//2,w//2).transpose(1,2).contiguous().view(-1,1,h,w)
    return res


def iwt(vres, levels=1):
    h = vres.size(2)
    w = vres.size(3)
    res = vres.view(-1,h//2,2,w//2).transpose(1,2).contiguous().view(-1,4,h//2,w//2).clone()
    if levels>1:
        res[:,:1] = iwt(res[:,:1], levels=levels-1)
    res = torch.nn.functional.conv_transpose2d(res, inv_filters[:,None], stride=2)
    filt_len = rec_lo.shape[-1]  
    pad = (2 * filt_len - 3) // 2      
    res = res[:,:,pad:-pad,pad:-pad]
    return res

class compute_Q_dwt(compute_Q):
    
    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.dwt = lambda x: wt(x)
        self.idwt = lambda x: iwt(x)

    def sample_hessian(self, img, sampler):
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device).type(torch.float32)
        img_dwt = self.dwt(img)
        img_rec = self.idwt(img_dwt)
        #print(torch.sum(torch.abs(img-img_rec)).item())
        # crop img to have the same size as Ya
        sampler = torch.from_numpy(sampler).unsqueeze(0).unsqueeze(0).to(device).type(torch.float32)
        # central padding to have the same size as img
        padl = img_dwt.shape[3]//2 - sampler.shape[3]//2
        padr = img_dwt.shape[3]//2 - sampler.shape[3]//2
        padt = img_dwt.shape[2]//2 - sampler.shape[2]//2
        padb = img_dwt.shape[2]//2 - sampler.shape[2]//2
        sampler = torch.nn.functional.pad(sampler, [padl, padr, padt, padb], mode='constant', value=0)

        # form img2 as img1 using each of the subbands
        func = lambda x: self.master_func(img_dwt, x)

        handle_out = torch.autograd.functional.hvp(func, img_dwt, sampler)[1]
        # remove the padding
        handle_out = handle_out[:,:,padt:-padb,padl:-padr]
        del img, sampler, img_dwt, img_rec
        torch.cuda.empty_cache()
        out_val = handle_out.cpu().detach().numpy().squeeze()
        return out_val


class compute_Q_ssim_dwt(compute_Q_dwt):

    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: ssim_func(self.idwt(x), self.idwt(y))


class compute_Q_msssim_dwt(compute_Q_dwt):

    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: ms_ssim_func(self.idwt(x), self.idwt(y))


class compute_Q_lpips_dwt(compute_Q_dwt):

    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: compute_LPIPS_gs(self.idwt(x), self.idwt(y))

    
class compute_Q_brisque_dwt(compute_Q_dwt):

    def __init__(self, true_N, n_cwd, N):
        super().__init__(true_N, n_cwd, N)
        self.master_func = lambda x, y: brisque_func(self.idwt(x), self.idwt(y))
