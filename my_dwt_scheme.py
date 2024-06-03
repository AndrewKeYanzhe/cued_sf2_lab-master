import numpy as np
from typing import Optional, Tuple
from cued_sf2_lab.laplacian_pyramid import rowdec, rowint, quantise, quant1, quant2, bpp
from cued_sf2_lab.dct import colxfm, dct_ii, regroup
from cued_sf2_lab.lbt import pot_ii, dct_ii
from cued_sf2_lab.dwt import dwt, idwt
from scipy import optimize
from cued_sf2_lab.jpeg import diagscan, dwtgroup, huffdflt, HuffmanTable, huffenc, huffgen, huffdes, runampl
from time import perf_counter

def compress_dwt(X: np.ndarray, n: int, log: bool = True) -> np.ndarray:
    img_size = (256, 256)
    Y = X.copy()
    for i in range(n):
        m = 256 // (2 ** i)
        Y[:m, :m] = dwt(Y[:m, :m])
    return Y

def decompress_dwt(Y: np.ndarray, n: int) -> np.ndarray:
    Yr = Y.copy()
    for i in range(n):
        m = 256 // (2 ** (n - i - 1))
        Yr[:m, :m] = idwt(Yr[:m, :m])
    return Yr

def estimate_entropy_dwt(Y: np.ndarray, n: int) -> int:
    dwtent = 0
    for i in range(n):
        m = 256 // (2 ** i)
        h = m // 2
        dwtent += bpp(Y[:h, h:m]) + bpp(Y[h:m, :h]) + bpp(Y[h:m, h:m])
    dwtent += bpp(Y[:m, :m])
    return dwtent

def constant_steps_dwt(n: int, step: float = 1.) -> np.ndarray:
    dwtstep = np.ones((3, n)) * step
    return np.concatenate((dwtstep, np.ones((3, 1))), axis=1)

def equal_mse_steps_dwt(n: int, initial: float = 1., ratio: float = 2., root2: bool = True) -> np.ndarray:
    if root2:
        const_ratio = np.logspace(start=n, stop=0, num=n, base=ratio) * initial
        dwtstep = np.stack((const_ratio, const_ratio, const_ratio * np.sqrt(2)))
        dwtstep = np.concatenate((dwtstep, np.ones((3, 1))), axis=1)
    else:
        dwtstep = np.array([np.ones((1, 3))[0] * initial * (0.5 ** i) for i in range(n + 1)]).T
    return dwtstep

def quantise_dwt(Y: np.ndarray, steps: np.ndarray, n: int, rise_ratio: Optional[float] = None) -> np.ndarray:
    Yq = np.zeros_like(Y)
    if rise_ratio is None:
        rise_ratio = 0.5
    for i in range(n):
        m = 256 // (2 ** i)
        h = m // 2
        s_tr = max(steps[0, i], 1)
        s_bl = max(steps[1, i], 1)
        s_br = max(steps[2, i], 1)
        Yq[:h, h:m] = quant(Y[:h, h:m], s_tr, s_tr * rise_ratio)
        Yq[h:m, :h] = quant(Y[h:m, :h], s_bl, s_bl * rise_ratio)
        Yq[h:m, h:m] = quant(Y[h:m, h:m], s_br, s_br * rise_ratio)
    m = 256 // (2 ** n)
    s_tr = max(steps[0, n], 1)
    Yq[:m, :m] = quant(Y[:m, :m], s_tr, s_tr * rise_ratio)
    return Yq.astype(int)

def inv_quantise_dwt(Y: np.ndarray, steps: np.ndarray, n: int, rise_ratio: Optional[float] = None) -> np.ndarray:
    Yq = np.zeros_like(Y)
    if rise_ratio is None:
        rise_ratio = 0.5
    for i in range(n):
        m = 256 // (2 ** i)
        h = m // 2
        s_tr = max(steps[0, i], 1)
        s_bl = max(steps[1, i], 1)
        s_br = max(steps[2, i], 1)
        Yq[:h, h:m] = inv_quant(Y[:h, h:m], s_tr, s_tr * rise_ratio)
        Yq[h:m, :h] = inv_quant(Y[h:m, :h], s_bl, s_bl * rise_ratio)
        Yq[h:m, h:m] = inv_quant(Y[h:m, h:m], s_br, s_br * rise_ratio)
    m = 256 // (2 ** n)
    s_tr = max(steps[0, n], 1)
    Yq[:m, :m] = inv_quant(Y[:m, :m], s_tr, s_tr * rise_ratio)
    return Yq.astype(int)

def encode_dwt(Y: np.ndarray, n: int, qstep: Optional[int] = None, M: Optional[int] = None, dcbits: int = 16, rise_ratio: Optional[float] = None, root2: bool = True) -> Tuple[np.ndarray, HuffmanTable]:
    if qstep is None:
        dwtsteps = constant_steps_dwt(n)
    else:
        dwtsteps = equal_mse_steps_dwt(n, qstep, root2=root2)
    Yq = quantise_dwt(Y, dwtsteps, n, rise_ratio=rise_ratio)
    Yq = dwtgroup(Yq, n)
    N = np.round(2 ** n)
    if M is None:
        M = N
    scan = diagscan(M)
    huffhist = np.zeros(16 ** 2)
    for r in range(0, Yq.shape[0], M):
        for c in range(0, Yq.shape[1], M):
            yq = Yq[r:r + M, c:c + M]
            if M > N:
                yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            dccoef = yqflat[0] + 2 ** (dcbits - 1)
            ra1 = runampl(yqflat[scan])
            huffhist += huffblockhist(ra1)
    vlc = []
    dhufftab = huffdes(huffhist)
    huffcode, ehuf = huffgen(dhufftab)

    for r in range(0, Yq.shape[0], M):
        for c in range(0, Yq.shape[1], M):
            yq = Yq[r:r + M, c:c + M]
            if M > N:
                yq = regroup(yq, N)
            yqflat = yq.flatten('F')
            dccoef = yqflat[0] + 2 ** (dcbits - 1)
            if dccoef > 2 ** dcbits:
                raise ValueError('DC coefficients too large for desired number of bits')
            vlc.append(np.array([[dccoef, dcbits]]))
            ra1 = runampl(yqflat[scan])
            vlc.append(huffencopt(ra1, ehuf))

    vlc = np.concatenate([np.zeros((0, 2), dtype=np.intp)] + vlc)

    return vlc, dhufftab

def decode_dwt(vlc: np.ndarray, n: int, qstep: Optional[int] = None, hufftab: Optional[HuffmanTable] = None, N: int = 8, M: int = 8, dcbits: int = 16, rise_ratio: Optional[float] = None, root2: bool = True) -> np.ndarray:
    N = np.round(2 ** n)
    if M is None:
        M = N

    if qstep is None:
        dwtsteps = constant_steps_dwt(n)
    else:
        dwtsteps = equal_mse_steps_dwt(n, qstep, root2=root2)

    scan = diagscan(M)

    huffstart = np.cumsum(np.block([0, hufftab.bits[:15]]))
    huffcode, ehuf = huffgen(hufftab)

    k = 2 ** np.arange(17)

    eob = ehuf[0]
    run16 = ehuf[15 * 16]
    i = 0
    Zq = np.zeros((256, 256))

    W, H = 256, 256
    for r in range(0, H, M):
        for c in range(0, W, M):
            yq = np.zeros(M ** 2)

            cf = 0
            if vlc[i, 1] != dcbits:
                raise ValueError('The bits for the DC coefficient does not agree with vlc table')

            yq[cf] = vlc[i, 0] - 2 ** (dcbits - 1)
            i += 1

            while vlc[i, 0] != eob[0]:
                run = 0

                while vlc[i, 0] == run16[0]:
                    run += 16
                    i += 1

                start = huffstart[vlc[i, 1] - 1]
                res = hufftab.huffval[start + vlc[i, 0] - huffcode[start]]
                run += res // 16
                cf += run + 1
                si = res % 16
                i += 1

                ampl = vlc[i, 0]
                thr = k[si - 1]
                yq[scan[cf - 1]] = ampl - (ampl < thr) * (2 * thr - 1)
                i += 1

            i += 1
            yq = yq.reshape((M, M)).T

            if M > N:
                yq = regroup(yq, M // N)

            Zq[r:r + M, c:c + M] = yq

    Zq = dwtgroup(Zq, -n)
    Z = inv_quantise_dwt(Zq, dwtsteps, n, rise_ratio=rise_ratio)
    return Z

def opt_encode_dwt(Y: np.ndarray, n: int, size_lim: int = 40960, M: int = 8, root2: bool = True, rise_ratio: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    def error(qstep: int) -> int:
        Z, h = encode_dwt(Y, n, qstep=qstep, M=M, rise_ratio=rise_ratio, root2=root2)
        size = Z[:, 1].sum()
        return np.sum((size - size_lim) ** 2)

    opt_step = optimize.minimize_scalar(error, method="bounded", bounds=(0.1, 64)).x
    vlc, hufftab = encode_dwt(Y, n, qstep=opt_step, M=M, rise_ratio=rise_ratio, root2=root2)
    return vlc, hufftab, opt_step

# -----quantisation------
def quant(x, step, rise1=None):
    """
    Quantise the matrix x using steps of width step.

    The result is the quantised integers Q. If rise1 is defined,
    the first step rises at rise1, otherwise it rises at step/2 to
    give a uniform quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        q = x.copy()
        return q
    if rise1 is None:
        rise = step/2.0
    else:
        rise = rise1
    # Quantise abs(x) to integer values, and incorporate sign(x)..
    temp = np.ceil((np.abs(x) - rise)/step)
    q = temp*(temp > 0)*np.sign(x)
    return q

def inv_quant(q, step, rise1=None):
    """
    Reconstruct matrix Y from quantised values q using steps of width step.

    The result is the reconstructed values. If rise1 is defined, the first
    step rises at rise1, otherwise it rises at step/2 to give a uniform
    quantiser with a step centred on zero.
    In any case the quantiser is symmetrical about zero.
    """
    if step <= 0:
        y = q.copy()
        return y
    if rise1 is None:
        rise = step/2.0
        return q * step
    else:
        rise = rise1
        # Reconstruct quantised values and incorporate sign(q).
        y = q * step + np.sign(q) * (rise - step/2.0)
        return y

# -----huffman-----
def huffblockhist(rsa: np.ndarray):
    """Create a huffman histogram for a single block"""
    if max(rsa[:, 1]) > 10:
        print("Warning: Size of value in run-amplitude "
              "code is too large for Huffman table")
    rsa[rsa[:, 1] > 10, 1:3] = [10, (2 ** 10) - 1]

    huffhist = np.zeros(16 ** 2) 
    for i in range(rsa.shape[0]):
        run = rsa[i, 0]
        # If run > 15, use repeated codes for 16 zeros.
        run, count16 = run % 16, run // 16
        huffhist[15 * 16] += count16
        # Code the run and size.
        # Got rid off + 1 to suit python indexing.
        code = run * 16 + rsa[i, 1]
        huffhist[code] += 1

    return huffhist


def huffencopt(rsa: np.ndarray, ehuf: np.ndarray
    ) -> np.ndarray:
    """
    Convert a run-length encoded stream to huffman coding.

    Parameters:
        rsa: run-length information as provided by `runampl`.
        ehuf: the huffman codes and their lengths

    Returns:
        vlc: Variable-length codewords, consisting of codewords in ``vlc[:,0]``
            and corresponding lengths in ``vlc[:,1]``.
    """
    if max(rsa[:, 1]) > 10:
        print("Warning: Size of value in run-amplitude "
              "code is too large for Huffman table")
    rsa[rsa[:, 1] > 10, 1:3] = [10, (2 ** 10) - 1]

    vlc = []
    for i in range(rsa.shape[0]):
        run = rsa[i, 0]
        # If run > 15, use repeated codes for 16 zeros.
        run, count16 = run % 16, run // 16
        vlc += [ehuf[15 * 16] for _ in range(count16)]
        # Code the run and size.
        code = run * 16 + rsa[i, 1]
        vlc.append(ehuf[code])
        # If size > 0, add in the remainder (which is not coded).
        if rsa[i, 1] > 0:
            vlc.append(rsa[i, [2, 1]])
    return np.array(vlc)
