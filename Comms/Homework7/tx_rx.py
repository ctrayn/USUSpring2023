import numpy as np
from random import random
import matplotlib.pyplot as plt
from differentiate import Differentiator

class TX:
    def __init__(self, num_bits, pulse, upsample_num) -> None:
        self.bits = [int(random() > 0.5) for _ in range(num_bits)]
        self.I = []
        self.Q = []
        self.I_shaped = []
        self.Q_shaped = []
        self.upsample_num = upsample_num
        self.pulse = pulse

        self.bits_to_IQ()
        self.I_up = self.upsample(self.I)
        self.Q_up = self.upsample(self.Q)
        self.shape_pulses()

    def bits_to_IQ(self):
        self.I = []
        self.Q = []
        for index in range(int(len(self.bits)/2)):
            self.I.append(-1 if self.bits[index * 2    ] == 0 else 1)
            self.Q.append(-1 if self.bits[index * 2 + 1] == 0 else 1)

    def upsample(self, signal):
        sig = np.zeros((len(signal)*self.upsample_num,1))
        sig[range(0,len(signal)*self.upsample_num,self.upsample_num)] = np.array(signal).reshape(len(signal),1)
        return sig[:,0]
    
    def shape_pulses(self):
        self.I_shaped = np.convolve(self.I_up, self.pulse)
        self.Q_shaped = np.convolve(self.Q_up, self.pulse)

    def plot_signal(self, file_name="tx_signal.png", format='png'):
        plt.figure()
        plt.subplot(2,1,1)
        plt.title("Transmitted Signal")
        plt.ylabel("I")
        plt.plot(self.I_shaped)
        plt.subplot(2,1,2)
        plt.ylabel("Q")
        plt.plot(self.Q_shaped)
        plt.savefig(file_name, format=format)

    def get_signal(self):
        return self.I_shaped, self.Q_shaped
        

class RX:
    def __init__(self, I, Q, sample_time, Lp, pulse, diff_filter_len, diff_T) -> None:
        self.differentiator = Differentiator(diff_T, diff_filter_len)
        self.max_constellation_val = 1
        
        # Matched Filter
        self.I_matched = self.matched_filter(I, pulse)
        self.Q_matched = self.matched_filter(Q, pulse)

        # Differentiating and Delaying
        self.differentiator.differentiate(self.I_matched)
        self.I_deriv, self.I_delay = self.differentiator.get_result()
        self.differentiator.differentiate(self.Q_matched)
        self.Q_deriv, self.Q_delay = self.differentiator.get_result()
        
        # Sampling and Normalizing
        self.I_delay_sample = self.sample_signal(self.I_delay, sample_time, Lp)
        self.Q_delay_sample = self.sample_signal(self.Q_delay, sample_time, Lp)
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

    def sample_signal(self, sig, sample_time, Lp=0):
        sampled = []
        # for idx in range(Lp,len(sig)-Lp,sample_time):
        for idx in range(0,len(sig) - Lp, sample_time):
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