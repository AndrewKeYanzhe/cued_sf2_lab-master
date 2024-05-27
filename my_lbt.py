import numpy as np
import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise
from cued_sf2_lab.laplacian_pyramid import bpp
from cued_sf2_lab.dct import dct_ii
from cued_sf2_lab.dct import regroup
from cued_sf2_lab.dct import colxfm
from cued_sf2_lab.lbt import pot_ii
from scipy.optimize import minimize_scalar

def lbt(X, N, s, refstep):
    """
    Parameters:
        X: input image
        N: N * N blocks
        s: scaling factor
        refstep: reference step for direct quantisation
    """ 
    Xq = quantise(X, refstep)

    C = dct_ii(N)
    Pf, Pr = pot_ii(N, s)
    opt_step = find_optimal_step_size(X, N, s, refstep)[0]

    t = np.s_[N//2:-N//2] 
    Xp = X.copy() 
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T

    Y = colxfm(colxfm(Xp, C).T, C).T        # dct
    Yq = quantise(Y, opt_step)              # quantisation
    Yr = regroup(Yq, N)/N                   # regroup
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)    # reconstruction

    Zp = Z.copy() 
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)

    print(f"RMS error: {np.std(Zp-X)}")

    tot_bits_comp = dctbpp(Yr, 16)          # always use dctbpp(Yr, 16)
    tot_bits_direct = bpp(Xq) * Xq.size
    compression_ratio = tot_bits_direct / tot_bits_comp
    print(f"Total compression bits: {tot_bits_comp}")
    print(f"Compression ratio: {compression_ratio} \n")

    fig, ax = plt.subplots()
    plot_image(Zp, ax=ax)
    ax.set(title=f"{N} x {N} block, s = {s}")

    return None

def dctbpp(Yr, N):
    total_bits = 0
    d, d = np.shape(Yr)
    step = d//N
    for i in range(0, d, step):
        for j in range(0, d, step):
            Ys = Yr[i:i+step, j:j+step] 
            bits = bpp(Ys) * Ys.size
            total_bits += bits 
    return total_bits

def find_optimal_step_size(X, N, s, refstep):
    """Find the optimal step size by minimising RMS error (matched RMS error)"""
    def mse_diff(opt_step):
            
        Xq = quantise(X, refstep)
        optimalMSE = np.std(X - Xq) # direct quantisation 

        C = dct_ii(N)
        Pf, Pr = pot_ii(N, s)

        t = np.s_[N//2:-N//2] 
        Xp = X.copy() 
        Xp[t,:] = colxfm(Xp[t,:], Pf)
        Xp[:,t] = colxfm(Xp[:,t].T, Pf).T

        Y = colxfm(colxfm(Xp, C).T, C).T        # dct
        Yq = quantise(Y, opt_step)              # quantisation
        Yr = regroup(Yq, N)/N                   # regroup
        Z = colxfm(colxfm(Yq.T, C.T).T, C.T)    # reconstruction

        Zp = Z.copy() 
        Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
        Zp[t,:] = colxfm(Zp[t,:], Pr.T)

        MSE = np.std(Zp - X)

        MSEdiff = abs(MSE - optimalMSE)
        return MSEdiff
    
    res = minimize_scalar(mse_diff, bounds=(1, 256), method='bounded')
    minMSE = res.fun  # minimum
    minstep = res.x  # minimizer
    return minstep, minMSE

def find_optimal_scaling_factor(X, N, refstep):
    """Find the optimal scaling factor by maximising the compression ratio"""
    scaling_factors = []
    compression_ratios = []
    for s in np.linspace(1, 2, 100):
        
        Xq = quantise(X, refstep)

        C = dct_ii(N)
        Pf, Pr = pot_ii(N, s)
        opt_step = find_optimal_step_size(X, N, s, refstep)[0]

        t = np.s_[N//2:-N//2] 
        Xp = X.copy() 
        Xp[t,:] = colxfm(Xp[t,:], Pf)
        Xp[:,t] = colxfm(Xp[:,t].T, Pf).T

        Y = colxfm(colxfm(Xp, C).T, C).T        # dct
        Yq = quantise(Y, opt_step)              # quantisation
        Yr = regroup(Yq, N)/N                   # regroup
        Z = colxfm(colxfm(Yq.T, C.T).T, C.T)    # reconstruction

        Zp = Z.copy() 
        Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
        Zp[t,:] = colxfm(Zp[t,:], Pr.T)

        tot_bits_comp = dctbpp(Yr, 16)
        tot_bits_direct = bpp(Xq) * Xq.size
        compression_ratio = tot_bits_direct / tot_bits_comp

        scaling_factors.append(s)
        compression_ratios.append(compression_ratio)

        indexmax = np.argmax(compression_ratios)
        optimal_comp_ratio = compression_ratios[indexmax]
        optimal_s = scaling_factors[indexmax]

        return optimal_s, optimal_comp_ratio