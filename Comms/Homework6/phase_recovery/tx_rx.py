import numpy as np

def upsample(pulses, num_up):
    sig = np.zeros((len(pulses)*num_up,1))
    sig[range(0,len(pulses)*num_up,num_up)] = np.array(pulses).reshape(len(pulses),1)
    return sig[:,0]

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

def slice_QPSK(x, y):
    a0 = -1 if x < 0 else 1
    a1 = -1 if y < 0 else 1
    return a0, a1