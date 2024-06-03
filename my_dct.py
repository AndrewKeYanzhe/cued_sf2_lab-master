import numpy as np
import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import plot_image
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.dct import colxfm


from typing import Tuple
from scipy.optimize import minimize_scalar

def dct(X, N, step_size, refstep=17, rise1_factor=1.0, plot=True):
    """
    Parameters:
        X: input image
        N: N * N blocks
        step_size: step size for quantisation
        refstep = 17: reference step size for direct quantisation, for dctbpp
        rise1_factor = 0.5: the point at which the first rise occurs on each side of the zero step, 
                            range: (0,2)
        plot = True: whether plot the recontructed image
    Returns:
        Yq: quantised dct image
        Yr: regrouped (after quantisation) dct image 
        Z: reconstructed dct image
        rms_error: rms error between reconstructed image Z and input image X
        tot_bits_comp: total bits needed for compression
    """ 
    
    Xq = quantise(X, refstep)

    C = dct_ii(N)
    # opt_step = find_optimal_step_size(X, refstep, C)[0]
    
    Y = colxfm(colxfm(X, C).T, C).T         # dct
    Yq = quantise(Y, step_size, rise1_factor*step_size)              # quantisation
    Yr = regroup(Yq, N)/N
    tot_bits_comp = dctbpp(Yr, N)
    tot_bits_direct =  bpp(Xq) * Xq.size

    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)    # reconstruction
    rms_error = np.std(Z - X)
    comp_ratio = tot_bits_direct / tot_bits_comp

    print(f"RMS error: {rms_error}")
    print(f"Compression ratio: {comp_ratio}")

    if plot:
        fig, ax = plt.subplots()
        plot_image(Z, ax=ax)
        ax.set(title=f"{N} x {N} block")

    return Yq, Yr, Z, rms_error, tot_bits_comp

def dctbpp(Yr, N):
    # Your code here
    total_bits = 0
    d, d = np.shape(Yr)
    step = d//N
    for i in range(0, d, step):
        for j in range(0, d, step):
            Ys = Yr[i:i+step, j:j+step] 
            bits = bpp(Ys) * Ys.size
            total_bits += bits 
    return total_bits

'''
def find_optimal_step_size(X, N, refstep):
    def mse_diff(opt_step):

        C = dct_ii(N)

        Xq = quantise(X, refstep)
        optimalMSE = np.std(X - Xq) # direct quantisation 

        Y = colxfm(colxfm(X, C).T, C).T       # dct
        Yq = quantise(Y, opt_step)           # quantisation
        Yr = regroup(Yq, N)/N                  # regroup
        Z = colxfm(colxfm(Yq.T, C.T).T, C.T)  # reconstruction
        MSE = np.std(Z - X)

        MSEdiff = abs(MSE - optimalMSE)
        return MSEdiff

    res = minimize_scalar(mse_diff, bounds=(1, 256), method='bounded')
    minMSE = res.fun  # minimum
    minstep = res.x  # minimizer
    return minstep, minMSE

    optimal_step, optimal_MSE = find_optimal_step_size(X, N, refstep)
    print("Optimal step size: ", optimal_step)
    print("Optimal MSE: ", optimal_MSE)
'''