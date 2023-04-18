import math
import numpy as np
from random import random
import matplotlib.pyplot as plt
from differentiate import Differentiator

class RX:
    def __init__(self, signal, sample_time, Lp, pulse, diff_filter_len, diff_T, Omega0) -> None:
        self.differentiator = Differentiator(diff_T, diff_filter_len)
        self.max_constellation_val = 1.15
        self.Omega0 = Omega0

        n = np.arange(len(signal))
        C =  math.sqrt(2)*np.cos(Omega0*n)
        S = -math.sqrt(2)*np.sin(Omega0*n)

        I = [signal[index] * C[index] for index in range(len(signal))]
        Q = [signal[index] * S[index] for index in range(len(signal))]
        
        # Matched Filter
        self.I_matched = self.matched_filter(I, pulse)
        self.Q_matched = self.matched_filter(Q, pulse)

        # Differentiating and Delaying
        self.differentiator.differentiate(self.I_matched)
        self.I_deriv, self.I_delay = self.differentiator.get_result()
        self.differentiator.differentiate(self.Q_matched)
        self.Q_deriv, self.Q_delay = self.differentiator.get_result()
        
        # Sampling and Normalizing
        delay_time = int((diff_filter_len - 1) / 2)
        self.I_delay_sample = self.sample_signal(self.I_delay, delay_time, sample_time, Lp)
        self.Q_delay_sample = self.sample_signal(self.Q_delay, delay_time, sample_time, Lp)
        self.I_delay_normal = self.normalize_amplitude(self.I_delay_sample, self.max_constellation_val)
        self.Q_delay_normal = self.normalize_amplitude(self.Q_delay_sample, self.max_constellation_val)

        self.I_deriv_sample = self.sample_signal(self.I_deriv, sample_time, Lp)
        self.Q_deriv_sample = self.sample_signal(self.Q_deriv, sample_time, Lp)
        self.I_deriv_normal = self.normalize_amplitude(self.I_deriv_sample, self.max_constellation_val)
        self.Q_deriv_normal = self.normalize_amplitude(self.Q_deriv_sample, self.max_constellation_val)

    def get_sampled_signal(self):
        return self.I_delay_normal, self.Q_delay_normal

    def get_sampled_deriv(self):
        return self.I_deriv_normal, self.Q_deriv_normal
    
    def plot_sampled_signal(self, file_name='rx_sampled.png', format='png'):
        plt.figure()
        plt.subplot(2,1,1)
        plt.title("Received Samples")
        plt.stem(self.I_delay_normal)
        plt.ylabel("I")
        plt.subplot(2,1,2)
        plt.ylabel("Q")
        plt.stem(self.Q_delay_normal)
        plt.savefig(file_name, format=format)

    def sample_signal(self, sig, delay_time, sample_time, Lp=0):
        sampled = []
        # for idx in range(Lp,len(sig)-Lp,sample_time):
        for idx in range(delay_time, len(sig) - Lp, sample_time):
            sampled.append(sig[idx])
        return sampled

    def normalize_amplitude(self, signal, max_constellation_val):
        max_sig = max(max(signal),abs(min(signal)))
        return [i * max_constellation_val / max_sig for i in signal]

    def slice_QPSK(self, x, y):
        """Slices a single QPSK point"""
        a0 = 0 if x < 0 else 1
        a1 = 0 if y < 0 else 1
        return [a0, a1]

    def matched_filter(self, signal, pulse):
        pulse_reversed = list(reversed(pulse))
        return np.convolve(signal, pulse_reversed)