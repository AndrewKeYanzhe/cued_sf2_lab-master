import numpy as np
import matplotlib.pyplot as plt
from cued_sf2_lab.dwt import dwt
from cued_sf2_lab.dwt import idwt
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import bpp
from typing import Tuple
from scipy.optimize import minimize_scalar

def dwt_equal_step_size(X, n, step_size, refstep=17, plot=True):
    """
    Parameters:
        X: input image
        n: number of levels
        step_size: step size for quantisation
        refstep = 17: reference step for direct quantisation
        plot = True: whether plot the recontructed image
    Returns:
        Yq: quantised dwt image
        Z: reconstructed dct image
        rms_error: rms error between reconstructed image Z and input image X
        tot_bits_comp: total bits needed for compression
    """
    # optimal_step = find_optimal_step_size(X, n, refstep)[0]
    # print(f"Optimal step size for {n} layers: {optimal_step}")

    dwtstep = np.ones((3, n+1)) * step_size 
    Y = nlevdwt(X, n)
    Yq, dwtent = quantdwt(Y, dwtstep)
    Z = nlevidwt(Yq, n)
    rms_error = np.std(Z-X)
    print(f"RMS error for {n} layers: {rms_error}")

    bits_comp = np.sum(dwtent)
    bits_direct = bpp(quantise(X, refstep)) * X.size
    comp_ratio = bits_direct / bits_comp
    print(f"Total compression bits needed for {n} layers: {bits_comp} bits")
    print(f"Compression ratio for {n} layers: {comp_ratio} \n")

    if plot:
        fig, ax = plt.subplots()
        plot_image(Z, ax=ax)
        ax.set(title=f"{n} levels")

    return Yq, Z, rms_error, bits_comp

def dwt_equal_mse(X, n, step_size, refstep=17):
    """
    Parameters:
        X: input image
        n: number of levels
        step_size: step size for quantisation
        refstep = 17: reference step for direct quantisation
    Returns:
        Yq: quantised dwt image
        Z: reconstructed dct image
        rms_error: rms error between reconstructed image Z and input image X
        tot_bits_comp: total bits needed for compression
    """

    # optimal_step = find_optimal_step_size_mse(X, n, refstep)[0]
    # print(f"Optimal step size for {n} layers: {optimal_step}")
    dwtstep = np.ones((3, n+1)) * step_size * find_step_ratios_mse(X, n)

    Y = nlevdwt(X, n)
    Yq, dwtent = quantdwt(Y, dwtstep)
    Z = nlevidwt(Yq, n)
    Z = post_process_image(Z, 0.2)
    rms_error = np.std(Z-X)
    print(f"RMS error for {n} layers: {rms_error}")

    bits_comp = np.sum(dwtent)
    bits_direct = bpp(quantise(X, refstep)) * X.size
    comp_ratio = bits_direct / bits_comp
    print(f"Total compression bits needed for {n} layers: {bits_comp} bits")
    print(f"Compression ratio for {3} layers: {comp_ratio} \n")

    fig, ax = plt.subplots()
    plot_image(Z, ax=ax)
    ax.set(title=f"{n} levels")

    return Yq, Z, rms_error, bits_comp

def nlevdwt(X, n):
    """Perform n-level DWT. """
    m = 256
    Y = dwt(X)
    for _ in range(n-1):
        m = m//2
        Y[:m, :m] = dwt(Y[:m, :m])
    return Y

def nlevidwt(Y, n):
    """Perform n-level inverse DWT. """
    m = 256
    m = m//(2**(n-1))
    Xr = Y.copy()
    Xr[:m, :m] = idwt(Xr[:m, :m])
    for _ in range(n-1):
        m = m*2
        Xr[:m, :m] = idwt(Xr[:m, :m])
    return Xr

def quantdwt(Y: np.ndarray, dwtstep: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parameters:
        Y: the output of `dwt(X, n)`
        dwtstep: an array of shape `(3, n+1)`
    Returns:
        Yq: the quantized version of `Y`
        dwtenc: an array of shape `(3, n+1)` containing the entropies
    """

    # Initialise
    n = dwtstep.shape[1] - 1
    Yq = Y.copy()
    dwtent = np.zeros_like(dwtstep)
    m_tot = np.shape(Y)[0]

    for i in range(n):
        m = m_tot // (2**(i+1))

        # Top right
        Yq[:m, m:2*m] = quantise(Y[:m, m:2*m], dwtstep[0, i])
        dwtent[0, i] = bpp(Yq[:m, m:2*m]) * Yq[:m, m:2*m].size

        # Bottom left
        Yq[m:2*m, :m] = quantise(Y[m:2*m, :m], dwtstep[1, i])
        dwtent[1, i] = bpp(Yq[m:2*m, :m]) * Yq[m:2*m, :m].size
        
        # Bottom right
        Yq[m:2*m, m:2*m] = quantise(Y[m:2*m, m:2*m], dwtstep[2, i])
        dwtent[2, i] = bpp(Yq[m:2*m, m:2*m]) * Yq[m:2*m, m:2*m].size

    # Final low-pass image
    m = m_tot // (2**n)
    Yq[:m, :m] = quantise(Y[:m, :m], dwtstep[0, n])
    dwtent[0, n] = bpp(Yq[:m, :m]) * Yq[:m, :m].size
    
    return Yq, dwtent

# ----- For Equal Step Size -----
# def find_optimal_step_size(X, n, step):
#     def mse_diff(opt_step):

#         Xq = quantise(X, step) # quantised X
#         optimalMSE = np.std(X - Xq)

#         Y = nlevdwt(X, n) # DWT
#         dwtstep = np.ones((3, n+1)) * opt_step  
#         Yq = quantdwt(Y, dwtstep)[0] # quantise
#         Z = nlevidwt(Yq, n) # iDWT
#         MSE = np.std(Z - X)

#         MSEdiff = abs(MSE - optimalMSE)
#         return MSEdiff

#     res = minimize_scalar(mse_diff, bounds=(1, 256), method='bounded')
#     minMSE = res.fun  # minimum
#     minstep = res.x  # minimizer
#     return minstep, minMSE

# ----- For Equal MSE -----
def find_step_ratios_mse(X, n): 
    impulse = 100.0 # impulse magnitude of 100
    ratios = np.empty((3,n+1))
    layers = np.linspace(1,n+1,n+1)
    d = X.shape[0] # assume row dimension = column dimension, i.e. square image

    for i in layers:
        i = int(i)
        if i == n+1:
            Xt = np.zeros((d,d))
            Yt = nlevdwt(Xt,n)
            mid = d//(2**i) 
            Yt[mid, mid] = impulse
            Xtr = nlevidwt(Yt, n)
            E = np.sum(Xtr**2.0)
            ratios[:,i-1] = 1 / (np.sqrt(E)/impulse)
        else:
            for k in range(3):
                Xt = np.zeros((d,d))
                Yt = nlevdwt(Xt, n)
                mid = d // (2**i)
                r = int(mid - (mid/2)* ((-1)**np.abs(k-2)))
                c = int(mid + ((-1)**k)*(mid/2))
                Yt[r , c] = impulse
                Xtr = nlevidwt(Yt,n)
                E = np.sum(Xtr**2.0)
                ratios[k,i-1] = 1 / (np.sqrt(E)/impulse)

    ratios /= ratios[0][0]  # Normalize w.r.t the first one
    return ratios

def find_optimal_step_size_mse(X, n, refstep):
    def mse_diff_mse(opt_step):

        Xq = quantise(X, refstep) # quantised X
        optimalMSE = np.std(X - Xq)

        Y = nlevdwt(X, n) # DWT
        dwtstep = find_step_ratios_mse(X, n) * opt_step  
        Yq = quantdwt(Y, dwtstep)[0] # quantise
        Z = nlevidwt(Yq, n) # iDWT
        MSE = np.std(Z - X)

        MSEdiff = abs(MSE - optimalMSE)
        return MSEdiff

    res = minimize_scalar(mse_diff_mse, bounds=(1, 256), method='bounded')
    minMSE = res.fun  # minimum
    minstep = res.x  # minimizer
    return minstep, minMSE


# ----- Below are the improvements -----

from scipy.ndimage import gaussian_filter
def post_process_image(Z, sigma=1):
    return gaussian_filter(Z, sigma)

