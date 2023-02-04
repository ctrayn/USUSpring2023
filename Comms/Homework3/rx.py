from math import log2, sqrt, cos, sin
import numpy as np

def recieved_to_IQ(received_sig, freq):
    I = [i * sqrt(2) *  cos(freq) for i in received_sig]
    Q = [i * sqrt(2) * -sin(freq) for i in received_sig]

    return I, Q

def matched_filter(signal, pulse):
    pulse_reversed = list(reversed(pulse))
    return np.convolve(signal, pulse)

def sample_signal(sig, sample_time, Lp=0):
    sampled = []
    for idx in range(Lp,len(sig)-Lp,sample_time):
    # for idx in range(2*Lp-1,len(sig),sample_time):
        sampled.append(sig[idx])
    
    return sampled

def normalize_amplitude(signal, max_constellation_val):
    max_sig = max(signal)
    return [i * max_constellation_val / max_sig for i in signal]

def slice(I, Q, LUT):
    """
    LUT should be a 2D array where the amplitudes of '00' are in index 0 of the list as a 2 element list
    Returns a list of bits
    """
    num_bits = log2(len(LUT))       # Determine the number of bits per symnol
    sliced = []
    for idx in range(len(I)):
        distances = []
        for look in range(len(LUT)):    # Get a list of all distances
            distances.append(sqrt((I[idx] - LUT[look][0])**2 + (Q[idx] - LUT[look][1])**2))
        dist = np.array(distances)
        min_index = np.where(dist == dist.min())[0][0] # Get the minimimum distance as an index
        sliced += [int(i) for i in bin(min_index)[2:].zfill(int(num_bits))] # convert integer to bits
    return sliced