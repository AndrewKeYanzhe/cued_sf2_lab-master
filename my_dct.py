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

def dct(X, N, refstep):
    Xq = quantise(X, refstep)
    C = dct_ii(N)
    opt_step = find_optimal_step_size(X, refstep, C)[0]
    Y = colxfm(colxfm(X, C).T, C).T         # dct
    Yq = quantise(Y, opt_step)              # quantisation
    Yr = regroup(Yq, N)/N
    tot_bits_comp = dctbpp(Yr, N)
    tot_bits_direct =  bpp(Xq) * Xq.size

    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)    # reconstruction

    print(f"RMS error: {np.std(Z-X)}")
    print("Compression ratio: ", tot_bits_direct / tot_bits_comp)
    fig, ax = plt.subplots()
    plot_image(Z, ax=ax)
    ax.set(title=f"{N} x {N} block")

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
     
    '''
    for i in range(N):
        for j in range(N):
            Ys = Yr[i:i+N, j:j+N] 

    '''
    return total_bits


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