import numpy as np
import matplotlib.pyplot as plt
from plots import plot_and_show
from math import sqrt, cos, sin

def serial_to_parallel(bits:list, num_bits_per_point:int=1) -> list:
    """bits are the bits to be split up into seperate chunks, num_bits_per_point is 2 for 4 QAM, 4 for 16QAM, etc..."""
    to_return = []
    if len(bits) % num_bits_per_point != 0:
        print(f"Error: length of bits ({len(bits)} doesn't match the given number of bits per point ({num_bits_per_point})")

    for idx in range(int(len(bits)/num_bits_per_point)):
        parallel = 0
        for n in range(num_bits_per_point):
            parallel = (parallel << 1) | bits[idx * num_bits_per_point + n]
        to_return.append(parallel)
    return to_return

def parallel_to_I_Q(parallel_bits, LUT):
    """
    parallel_bits is a list of lists containing the parallel value ([0, 1, 3, 2, 0, 3, 1, ...])
    LUT is a Lookup Up Table where each is a list with len == 2 for I and Q, and index 0 = [I_0, Q_0] and index 1 = [I_1, Q_1], etc...
    """
    I = []
    Q = []
    for num in range(len(parallel_bits)):
        I.append(LUT[parallel_bits[num]][0])
        Q.append(LUT[parallel_bits[num]][1])

    return I, Q

def upsample(pulses, num_up):
    sig = np.zeros((len(pulses)*num_up,1))
    sig[range(0,len(pulses)*num_up,num_up)] = np.array(pulses).reshape(len(pulses),1)
    return sig[:,0]

def IQ_to_sig(I, Q, pulse:list, freq):
    I_upsampled = upsample(I)
    Q_upsampled = upsample(Q)

    plot_and_show(I_upsampled, title="I upsampled")
    plot_and_show(Q_upsampled, title="Q_upsampled")

    I_shaped = np.convolve(I_upsampled, pulse)
    Q_shaped = np.convolve(Q_upsampled, pulse)

    plot_and_show(I_shaped, title="I shaped")
    plot_and_show(Q_shaped, title="Q_shaped")

    I_sig = [i * sqrt(2) *  cos(freq) for i in I_shaped]
    Q_sig = [q * sqrt(2) * -sin(freq) for q in Q_shaped]

    plot_and_show(I_sig, title="I signal")
    plot_and_show(Q_sig, title="Q signal")

    soft = [I_sig[i] + Q_sig[i] for i in range(len(I_sig))]

    plot_and_show(soft, title="S(t)")

    return soft