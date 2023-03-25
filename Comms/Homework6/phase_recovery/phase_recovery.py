#!/bin/python3
import differential
from tx_rx import *
from pulses import srrc1
from random import random
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin

K = 1
N = 100
Lp = 60
Ts = 1
alpha = 1
pulse = srrc1(alpha, N, Lp, Ts)

########################
# Transmit
########################

num_bits = 100
            # Every offset    + Random bits
sent_bits = [1,1,1,0,0,1,0,0] + [int(random() > 0.5) for _ in range(num_bits)]

I, Q = differential.encoder(sent_bits)

I_up = upsample(I, N)
Q_up = upsample(Q, N)

I_shaped = np.convolve(I_up, pulse)
Q_shaped = np.convolve(Q_up, pulse)

plt.figure()
plt.subplot(2,1,1)
plt.title("I&Q")
plt.ylabel("I")
plt.plot(I_shaped)
plt.subplot(2,1,2)
plt.ylabel("Q")
plt.plot(Q_shaped)
plt.savefig('Shaped_Pulses.png', format='png')


########################
# Receiver
########################

I_received = [I + np.random.normal(0,1) for I in I_shaped] #Add signal
Q_received = [Q + np.random.normal(0,1) for Q in Q_shaped]

plt.figure()
plt.subplot(2,1,1)
plt.title("Noisy Signal")
plt.ylabel("I")
plt.plot(I_received)
plt.subplot(2,1,2)
plt.ylabel("Q")
plt.plot(Q_received)
plt.savefig("NoisyFigure.png", format='png')

xt = matched_filter(I_received, pulse)
yt = matched_filter(Q_received, pulse)

xt_sampled = sample_signal(xt, N, Lp)
yt_sampled = sample_signal(yt, N, Lp)

xt_normal = normalize_amplitude(xt_sampled, 1)
yt_normal = normalize_amplitude(yt_sampled, 1)

plt.figure()
plt.subplot(2,1,1)
plt.title("X&Y")
plt.ylabel("X")
plt.stem(xt_normal)
plt.subplot(2,1,2)
plt.ylabel("Y")
plt.stem(yt_normal)
plt.savefig("sampled_X_Y.png", format='png')

########################
# Rotation & Decision & PLL
########################

theta = 0
for index in range(len(xt_normal)):
    a0, a1 = slice_QPSK(xt_normal[index], yt_normal[index])
    theta = PLL(xt_normal[index], yt_normal[index], theta)
    x_prime = K * ((a0 * cos(theta)) - (a1 * sin(theta)))

xt_prime = xt_normal
yt_prime = yt_normal

########################
# Points to bits
########################
x_decide = [0 if x < 0 else 1 for x in xt_prime]
y_decide = [0 if y < 0 else 1 for y in yt_prime]

########################
# Decoder
########################
r_bits = differential.decoder(x_decide, y_decide)

print(f"Do the bits match? {'Yes' if r_bits == sent_bits else 'No'}")
print("Sent bits")
print(sent_bits)
print("Received bits")
print(r_bits)

