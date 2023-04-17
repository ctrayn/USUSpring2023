import math
import numpy as np
import matplotlib.pyplot as plt
from tx_rx import *
from pulses import srrc1, slice_LUT

input_file = 'data/test_2023'

Ts = 1 # Symbol period
N = 4
Lp = 12
alpha = 0.7
diff_filter_len = 11
diff_T = 1
pulse = srrc1(alpha,N,Lp)
Omega0 = math.pi/2

with open(input_file, 'r') as in_file:
    input_signal = [float(line) for line in in_file.readlines()]
    # print(input_signal)

n_rows = input_signal.pop(0)
n_cols = input_signal.pop(0)

print(n_rows)
print(n_cols)

rx = RX(
    signal=input_signal,
    sample_time=N,
    Lp=Lp,
    pulse=pulse,
    diff_filter_len=diff_filter_len,
    diff_T=diff_T,
    Omega0=Omega0)
I, Q = rx.get_sampled_signal()

plt.figure()
plt.scatter(I, Q)
plt.savefig("images/test_2023_IQ.png", format='png')

bits = []
for index in range(3):
    bits.append(slice_LUT(I[index], Q[index]))
    print(bits[-1])
