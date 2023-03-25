#!/bin/python3
import differential
from tx_rx import *
from pll import PLL
from pulses import srrc1
from random import random
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin, atan2

K = 1
N = 100
Lp = 100
Ts = 1
alpha = 1
pulse = srrc1(alpha, N, Lp, Ts)

########################
# Transmit
########################

num_bits = 1000
            # Every offset    + Random bits
sent_bits = [1,1,1,0,0,1,0,0] + [int(random() > 0.5) for _ in range(num_bits)]

I, Q = differential.encoder(sent_bits)

I_up = upsample(I, N)
Q_up = upsample(Q, N)

I_shaped = np.convolve(I_up, pulse)
Q_shaped = np.convolve(Q_up, pulse)

# plt.figure()
# plt.subplot(2,1,1)
# plt.title("Transmitted Signal")
# plt.ylabel("I")
# plt.plot(I_shaped)
# plt.subplot(2,1,2)
# plt.ylabel("Q")
# plt.plot(Q_shaped)


########################
# Receiver
########################

I_received = [I for I in I_shaped] #Add signal
Q_received = [Q for Q in Q_shaped]

# I_received = [I + np.random.normal(0,1) for I in I_shaped] #Add signal
# Q_received = [Q + np.random.normal(0,1) for Q in Q_shaped]

# plt.figure()
# plt.subplot(2,1,1)
# plt.title("Noisy Signal")
# plt.ylabel("I")
# plt.plot(I_received)
# plt.subplot(2,1,2)
# plt.ylabel("Q")
# plt.plot(Q_received)

xt = matched_filter(I_received, pulse)
yt = matched_filter(Q_received, pulse)

xt_sampled = sample_signal(xt, N, Lp)
yt_sampled = sample_signal(yt, N, Lp)

xt_normal = normalize_amplitude(xt_sampled, 1)
yt_normal = normalize_amplitude(yt_sampled, 1)

# plt.figure()
# plt.subplot(2,1,1)
# plt.title("Sampled Signal")
# plt.ylabel("X")
# plt.stem(xt_normal)
# plt.subplot(2,1,2)
# plt.ylabel("Y")
# plt.stem(yt_normal)

plt.figure()
plt.scatter(xt_normal, yt_normal)
plt.title("Recieved I, Q")
plt.xlabel("I")
plt.ylabel("Q")

########################
# Rotation & Decision & PLL
########################

theta = 0
xt_prime = []
yt_prime = []
pll = PLL()
theta_hats = []
delta_thetas = []
for index in range(len(xt_normal)):
    a0, a1 = slice_QPSK(xt_normal[index], yt_normal[index])
    # theta = atan2(xt_normal[index], yt_normal[index])
    # theta_hat = pll.pll(theta)
    theta_hat = pll.pll(xt_normal[index], yt_normal[index], a0, a1)
    theta_hats.append(theta_hat)
    delta_theta = theta_hat
    delta_thetas.append(delta_theta)
    xt_prime.append(K * ((a0 * cos(delta_theta)) - (a1 * sin(delta_theta))))
    yt_prime.append(K * ((a0 * sin(delta_theta)) + (a1 * cos(delta_theta))))

plt.figure()
plt.subplot(2,1,1)
plt.title("Phase Locked Signal")
plt.ylabel("X")
plt.stem(xt_prime)
plt.subplot(2,1,2)
plt.ylabel("Y")
plt.stem(yt_prime)

plt.figure()
plt.scatter(xt_prime, yt_prime)
plt.title("IQ PLL out")
plt.xlabel("I")
plt.ylabel("Q")

plt.figure()
plt.plot(theta_hats)
plt.title("Theta Hat")
plt.figure()
plt.plot(delta_thetas)
plt.title("Delta Theta")

plt.figure()
plt.plot(pll.error_signal)
plt.title("Error Signal")

########################
# Points to bits
########################
x_decide = [0 if x < 0 else 1 for x in xt_prime]
y_decide = [0 if y < 0 else 1 for y in yt_prime]

########################
# Decoder
########################
r_bits = differential.decoder(x_decide, y_decide)


########################
# Comparisons
########################

yes = '\033[32mYes\033[0m'
no = '\033[31mNo\033[0m'

print(f"Do the bits match? {yes if r_bits == sent_bits else no}")
missed_count = 0
for index in range(len(sent_bits)):
    if sent_bits[index] != r_bits[index]:
        missed_count += 1
print(f"Error rate {missed_count / len(sent_bits)}")


plt.show()
