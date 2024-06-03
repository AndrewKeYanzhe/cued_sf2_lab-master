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

def lbt(X, N, s, step_size, refstep=17, rise1_factor=1.0, plot=True):
    """
    Parameters:
        X: input image
        N: N * N blocks
        s: scaling factor
        step_size: step size for quantisation
        refstep = 17: reference step size for direct quantisation, for dctbpp
        rise1_factor = 0.5: the point at which the first rise occurs on each side of the zero step, 
                            range: (0,2)
        plot = True: whether plot the recontructed image
    Returns:
        Yq: quantised lbt image
        Yr: regrouped (after quantisation) lbt image 
        Z: reconstructed dct image
        rms_error: rms error between reconstructed image Z and input image X
        tot_bits_comp: total bits needed for compression
    """ 
    Xq = quantise(X, refstep)

    C = dct_ii(N)
    Pf, Pr = pot_ii(N, s)
    # opt_step = find_optimal_step_size(X, N, s, refstep)[0]

    t = np.s_[N//2:-N//2] 
    Xp = X.copy() 
    Xp[t,:] = colxfm(Xp[t,:], Pf)
    Xp[:,t] = colxfm(Xp[:,t].T, Pf).T

    Y = colxfm(colxfm(Xp, C).T, C).T        # dct
    # Yq = quantise(Y, opt_step)              # quantisation
    Yq = quantise(Y, step_size, rise1_factor*step_size)              # quantisation
    Yr = regroup(Yq, N)/N                   # regroup
    Z = colxfm(colxfm(Yq.T, C.T).T, C.T)    # reconstruction

    Zp = Z.copy() 
    Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
    Zp[t,:] = colxfm(Zp[t,:], Pr.T)

    rms_error = np.std(Zp-X)
    print(f"RMS error: {rms_error}")

    tot_bits_comp = dctbpp(Yr, 16)          # always use dctbpp(Yr, 16)
    tot_bits_direct = bpp(Xq) * Xq.size
    compression_ratio = tot_bits_direct / tot_bits_comp
    print(f"Total compression bits: {tot_bits_comp}")
    print(f"Compression ratio: {compression_ratio} \n")

    if plot:
        fig, ax = plt.subplots()
        plot_image(Zp, ax=ax)
        ax.set(title=f"{N} x {N} block, s = {s}, step_size = {step_size}")

    return Yq, Yr, Z, rms_error, tot_bits_comp

# def lbt(X, N, s, step_size, refstep=17, plot=True):

#     # your code here
#     # N=4
#     print("N",N)


#     C8 = dct_ii(N) #note that c8 is labelled wrongly here but referencing the right matrix

#     # decimal_range_stepsize = np.arange(20, 30, 0.1)
#     # decimal_range_s = np.arange(1, 2, 0.01)



#     # optimal_s = -1
#     # highest_compression = -1


#     Pf, Pr = pot_ii(N,s)

#     t = np.s_[N//2:-N//2]  # N is the DCT size, I is the image size
#     Xp = X.copy()  # copy the non-transformed edges directly from X
#     Xp[t,:] = colxfm(Xp[t,:], Pf)
#     Xp[:,t] = colxfm(Xp[:,t].T, Pf).T

#     Y = colxfm(colxfm(Xp, C8).T, C8).T


#     # optimal_step = -1
#     # smallest_diff = 10e15

#     # def f(step_size):
#     #     Y_q_test = quantise(Y,step_size)
#     #     Z = colxfm(colxfm(Y_q_test.T, C8.T).T, C8.T)

#     #     Zp = Z.copy()  #copy the non-transformed edges directly from Z
#     #     Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
#     #     Zp[t,:] = colxfm(Zp[t,:], Pr.T)

#     #     return abs(np.std(X-Xq)-np.std(X-Zp))


#     # for i in decimal_range_stepsize:
#     #     # print(i,"             ",abs(f(i)))
#     #     if f(i) < smallest_diff:
#     #         smallest_diff = f(i)
#     #         optimal_step = i

#     # print("optimal_step",optimal_step)

#     optimal_step = step_size
#     Y_q = quantise(Y,optimal_step)


#     Yr=regroup(Y_q, N)/N  #Yr is from quantised
#     print("Total bits using dctbpp: {:,}".format(int(dctbpp(Yr, 16))))

#     Xq = quantise(X,17)
#     X_q_total_bits = bpp(quantise(X,17))*X.shape[0]*X.shape[1]

#     compression_ratio = X_q_total_bits/(dctbpp(Yr,16)) #Yr is quantised. this is variable coding N=4 here
#     print("compression ratio",compression_ratio)



#     Z = colxfm(colxfm(Y_q.T, C8.T).T, C8.T)
#     # Z = colxfm(colxfm(Y.T, C8.T).T, C8.T) #dont quantise for now



#     Zp = Z.copy()  #copy the non-transformed edges directly from Z
#     Zp[:,t] = colxfm(Zp[:,t].T, Pr.T).T
#     Zp[t,:] = colxfm(Zp[t,:], Pr.T)

#     Z_4 = Zp

#     # compression_ratio_dict[N] = compression_ratio

#     if plot:
#         fig, ax = plt.subplots()
#         plot_image(Zp, ax=ax)
#         ax.set(title=f"{N} x {N} block, s = {s}")

#     return Y_q, Yr



def lbt_reconstruct(Yq):
    return colxfm(colxfm(Yq.T, C.T).T, C.T) 

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

def find_optimal_step_size(X, N, s, refstep=17):
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