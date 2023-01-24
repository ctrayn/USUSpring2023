from numpy import convolve, reshape
from matplotlib import figure
from matplotlib.pyplot import plot, title
from math import ceil, floor
from random import random
from srrc import srrc1

# test eye diagram plots
alpha = 0.2                     # excess bandwidth
N = 11                          # samples per symbol
Lp = 50                         # SRRC truncation length (samples)
Ts = 1                          # symbol time
T = Ts/N                        # sample time
srrcout = srrc1(alpha,N,Lp,Ts)  # get the pulse

a = 1                           # PAM signal amplitude
LUT1d = [-1, 1]                 # 1-dimensional lookup table
Nsym = 1000                     # number of symbols
bits = (random(1,Nsym) > 0.5) + 0
ampa = LUT1d(bits + 1)          # look up the amplitudes (+1 is for 1-based indexing)
upsampled = [0] * (N*Nsym)      # space for upsampled pulses
for i in range(0,N*Nsym,N):     # make the upsampled pulses
    upsampled[i] = ampa
s = convolve(upsampled,srrcout)
x = convolve(s,srrcout)         # matched filter
figure(1)
plot(s)
figure(2)
# plot(x);
plot(x[1:N*20+1])

for i in range(1,10):
    plot([2*Lp + i*N + 1, 2*Lp + i*N + 1],[-2,2])

xn = x.size()
Nsymtoss = 2 * ceil(Lp/N)
# throw away tails at end
offset = 2*Lp + 1 + floor(N/2) + 1
print(f'l1={x[offset:xn - Nsymtoss*N].size()}\n')

nc = floor((xn - offset - Nsymtoss*N + 1)/N)    # number of symbols of signal to plot

xc = reshape(x[offset: offset + nc*N - 1],N,nc)

figure(3)
plot([i for i in range(-floor(N/2),floor(N/2))],xc,'b')
title(f'SRRC L_p={Lp} N={N}')