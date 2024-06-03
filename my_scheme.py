import numpy as np
import matplotlib.pyplot as plt
from cued_sf2_lab.familiarisation import load_mat_img, plot_image
from cued_sf2_lab.laplacian_pyramid import quantise, bpp
from cued_sf2_lab.dct import dct_ii, colxfm, regroup
from cued_sf2_lab.lbt import pot_ii
from cued_sf2_lab.dwt import dwt, idwt 
from my_lbt import *
from my_dwt import *

def lbt_compression(X, rise_ratio = 0.5):
    #Direct quantised
    X_quantised = quantise(X, 17)
    bppXq = bpp(X_quantised)
    tot_bits_direct = bppXq * X_quantised.size

    s_values=[]
    CRs=[] 
    N = 8
    C = dct_ii(N) # s optimisation using 8x8 dct
    for i in np.arange(1, 2, 0.1):
        s_values.append(i)
        Pf, Pr = pot_ii(N, i) # filter with different s

        #pre filtering
        t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
        Xp = X.copy()  # copy the non-transformed edges directly from X
        Xp[t,:] = colxfm(Xp[t,:], Pf)
        Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
        #dct
        Y = colxfm(colxfm(Xp, C).T, C).T
        step_size = find_optimal_step_size(X, N, i)[0]
        Yq = quantise(Y, step_size, step_size*rise_ratio)
        Yr = regroup(Yq, N)/N

        CRs.append(tot_bits_direct/dctbpp(Yr, N))

    OptimalS = s_values[np.argmax(CRs)]
    OptimalS = round(OptimalS, 3)
    OptimalCR = CRs[np.argmax(CRs)]
    OptimalCR = round(OptimalCR, 3)
    Pf, Pr = pot_ii(N, OptimalS) # filter with different s

    # pre filtering
    t = np.s_[N//2:-N//2] 
    Xp = X.copy() 
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T
    # dct
    Y = colxfm(colxfm(Xp, C).T, C).T
    step_size = find_optimal_step_size(X, N, i)[0]
    Yq = quantise(Y, step_size, step_size*rise_ratio)
    Yr = regroup(Yq, N)/N
    # inverse dct
    Zq = colxfm(colxfm(Yq.T, C.T).T, C.T)
    # post filtering
    Zqp = Zq.copy()  
    Zqp[:,t] = colxfm(Zqp[:,t].T, Pr.T).T
    Zqp[t,:] = colxfm(Zqp[t,:], Pr.T)
    print(f'Optimal CR after LBT is:{OptimalCR} at s = {OptimalS}' )
    # rms_error = np.std(Zqp - X)
    print(f"Total bits needed after LBT: {dctbpp(Yr, N)}")
    return OptimalCR, Zqp, Yr

def dwt_compression(X, rise_ratio=0.5):
    #Direct quantised
    X_quantised = quantise(X, 17)
    bppXq = bpp(X_quantised)
    tot_bits_direct = bppXq * X_quantised.size
    CRs = []
    #optimised n 
    n_values = [1,2]
    for n in n_values:
        Y = nlevdwt(X, n)
        dwtstep = np.full((3,n+1), find_optimal_step_size_mse(X, n, 0.1)[0])
        Yq, dwtent = quantdwt(Y,dwtstep)     
        CRs.append(np.sum(dwtent))
    Optimaln = n_values[np.argmax(CRs)]
    OptimalCR = CRs[np.argmax(CRs)]
    Y = nlevdwt(X, Optimaln)
    dwtstep = np.full((3,Optimaln + 1), find_optimal_step_size_mse(X, Optimaln, 0.1)[0])
    Yq, dwtent= quantdwt(Y, dwtstep)
    Zq = nlevidwt(Yq, Optimaln)
    return OptimalCR, Zq, Yq

def compression_scheme(X, rise_ratio=0.5):
    print(f"Rise ratio = {rise_ratio} (default = 0.5)")

    # Quantize the original image
    X_quantised = quantise(X, 17)
    bppXq = bpp(X_quantised)
    TbeXq = bppXq * X_quantised.size

    # Perform LBT compression
    Optimal_CR_lbt, Zqp, Yr = lbt_compression(X, rise_ratio=rise_ratio)

    # Perform DWT compression on each 32x32 block
    block_size = 32
    num_blocks = 8
    gaps = np.linspace(0, X.shape[0], num_blocks + 1, dtype=int)
    for i in range(num_blocks):
        for j in range(num_blocks):
            block_start_row, block_end_row = gaps[i], gaps[i + 1]
            block_start_col, block_end_col = gaps[j], gaps[j + 1]
            Yr_block = Yr[block_start_row:block_end_row, block_start_col:block_end_col]
            _, _, Yrdwt = dwt_compression(Yr_block, rise_ratio=rise_ratio)
            Yr[block_start_row:block_end_row, block_start_col:block_end_col] = Yrdwt

    # Calculate the total coded bits after DWT compression
    total_coded_bits = dctbpp(Yr, 8)
    print(f"Total bits needed after DWT: {total_coded_bits}")

    # Calculate the overall compression ratio
    overall_CR = round(TbeXq / total_coded_bits, 3)
    print(f"Optimal Overall CR is: {overall_CR}")

    return Yr

