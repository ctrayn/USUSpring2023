import numpy as np
from numpy.random import rand
from plots import plot_and_show

def get_filter_output(pulse_func, bits:list=None, LUT=np.array([-1,1]), alpha:float=1.0, N:int=11, Lp:int=60, Ts:float=1, Nsym:int=100):
    T = Ts/N
    srrcout = pulse_func(alpha,N,Lp,Ts); # get the square-root raised cosine pulse

    pam_len = int(len(LUT)/2)
    bit_idx = []
    if not bits:
        bits = (rand(Nsym*(pam_len))> 0.5).astype(int) # generate random bits {0,1}
    for idx in range(Nsym):   # Convert from 2 bits to one number
        sum = 0
        for bit in range(pam_len):
            sum = sum << 1 | bits[idx * pam_len + bit]
        bit_idx.append(sum)
    ampa = LUT[bit_idx] # map the bits to {+1,-1} values
    upsampled = np.zeros((N*Nsym,1))
    upsampled[range(0,N*Nsym,N)] = ampa.reshape(Nsym,1)
    s = np.convolve(upsampled.reshape((N*Nsym,)),srrcout) # the transmitted signal
    return s

def plot_trajectory(pulse_fun, alpha, LUT, Nsym, show_plot=True):
    It = get_filter_output(pulse_fun, alpha=alpha, LUT=LUT, Nsym=Nsym)
    Qt = get_filter_output(pulse_fun, alpha=alpha, LUT=LUT, Nsym=Nsym)
    plot_and_show(y=Qt, x=It, title=f"{len(LUT)**2}QAM Alpha={alpha}", xlabel="I(t)", ylabel="Q(t)")
