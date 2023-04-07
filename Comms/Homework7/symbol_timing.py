from numpy import fft, convolve, sin, cos, pi
from differentiate import Differentiator
import matplotlib.pyplot as plt
from pulses import *
from tx_rx import *
from random import random

K = 1
N = 100
Lp = 100
num_bits = 10
Ts = 1
alpha = 1
pulse = srrc1(alpha, N, Lp, Ts)

########################
# Transmit
########################

tx = TX(num_bits, pulse, N)
tx.plot_signal()
I_sent, Q_sent = tx.get_signal()


########################
# Receive
########################

diff_filter_len = 11
diff_T = 1

rx = RX(I_sent, Q_sent, 25, Lp, pulse, diff_filter_len, diff_T)
rx.plot_sampled_signal()

