import math
import numpy as np
import matplotlib.pyplot as plt
from tx_rx import *
from pulses import srrc1, slice_LUT, slice_LUT_get_symbols
from ted import TED
from pll import PLL
from interpolator import CubicInterpolator

input_file = 'sim3_2023'

Ts = 1 # Symbol period
N = 4
Lp = 12
alpha = 0.7
diff_filter_len = 11
diff_T = 1
pulse = srrc1(alpha,N,Lp)
Omega0 = math.pi/2

with open('data/' + input_file, 'r') as in_file:
    input_signal = [float(line) for line in in_file.readlines()]
    # print(input_signal)

rx = RX(
    signal=input_signal,
    # sample_time=N,
    sample_time=N,
    Lp=Lp,
    pulse=pulse,
    diff_filter_len=diff_filter_len,
    diff_T=diff_T,
    Omega0=Omega0)
I, Q = rx.get_sampled_signal()
Ip, Qp = rx.get_sampled_deriv()

plt.figure()
plt.stem(I)
# plt.xlim([0, 500])
plt.title("I")
plt.savefig(f'images/{input_file}_I.png', format='png')

assert len(I) == len(Q) == len(Ip) == len(Qp), f"RX Signal and Derivitive are not the same length {len(I)} {len(Q)} {len(Ip)} {len(Qp)}"

plt.figure()
plt.scatter(I, Q)
plt.title(f'{input_file}')
plt.savefig(f"images/{input_file}_IQ_pre_tedandpll.png", format='png')

##############################
# Tracking
##############################

# Interpolation
start_sample = 94
num_samples = 4
K0 = 1
Kp = 0.23
ted = TED(K0=K0, KP=Kp, num_samples=num_samples)
I_int = CubicInterpolator()
Q_int = CubicInterpolator()
Ip_int = CubicInterpolator()
Qp_int = CubicInterpolator()
I_results = [0]
Q_results = [0]
# Ip_results = [0]
# Qp_results = [0]
# mus = []

# PLL
pll = PLL()
theta = 0
K = 1
xt_prime = []
yt_prime = []
theta_hats = []
delta_thetas = []
for idx in range(start_sample, len(I)):
    # New samples
    # I_int.new_sample(I[idx])
    # Q_int.new_sample(Q[idx])
    # Ip_int.new_sample(Ip[idx])
    # Qp_int.new_sample(Qp[idx])

    # mu = ted.timing_error(I_results[-1], Q_results[-1], Ip_results[-1], Qp_results[-1])
    # mus.append(mu)
    # if ted.strobe:
    #     I_int.interpolate(mu); Q_int.interpolate(mu); Ip_int.interpolate(mu); Qp_int.interpolate(mu)
    #     # I_int.plot(f"I_int{idx-num_samples+1}-{idx}")
    #     I_results.append(I_int.get_result())
    #     Q_results.append(Q_int.get_result())
    #     Ip_results.append(Ip_int.get_result())
    #     Qp_results.append(Qp_int.get_result())

    a0, a1 = slice_LUT_get_symbols(I[idx], Q[idx])
    theta_hat = pll.pll(I[idx], Q[idx], a0, a1)
    theta_hats.append(theta_hat)
    # delta_thetas.append(theta_hat)
    I_results.append(K * ((a0 * np.cos(theta_hat)) - (a1 * np.sin(theta_hat))))
    Q_results.append(K * ((a0 * np.sin(theta_hat)) + (a1 * np.cos(theta_hat))))


# I_results.pop(0)
# Q_results.pop(0)
# Ip_results.pop(0)
# Qp_results.pop(0)

bits = []
for I, Q in zip(I_results, Q_results):
    bits.append(slice_LUT(I, Q))

print(f"Length of bits {len(bits)}")

# Interpolation output figures
# plt.figure()
# plt.plot(mus)
# plt.title("Mu(k)")
# plt.savefig(f"images/mus_{input_file}.png", format='png')

# plt.figure()
# plt.plot(ted.es)
# plt.title("Filter Error")
# plt.savefig(f"images/e_{input_file}.png", format='png')

# PLL output figures
plt.figure()
plt.plot(pll.error_signal)
plt.title("PLL Error Signal")
plt.savefig(f'images/pll_error_{input_file}.png', format='png')

plt.figure()
plt.plot(theta_hats)
plt.title("Theta Hat")
plt.savefig(f'images/pll_theta_hat_{input_file}.png', format='png')

# General output figures
plt.figure()
plt.scatter(I_results, Q_results)
plt.title("Recovered IQ")
plt.savefig(f"images/recovered_IQ_{input_file}.png", format='png')

##############################
# Bits to Image
##############################

n_rows = bits.pop(0)
n_cols = bits.pop(0)
# FIXME: Remove the first 23 points, should be the row and colum information
for _ in range(23):
    bits.pop(0)

print(f'Rows {n_rows}')
print(f'Cols {n_cols}')

unique_word = [162, 29, 92, 47, 16, 112, 63, 234, 50, 7, 15, 211, 109, 124, 239, 255, 243, 134, 119, 40, 134, 158, 182, 0, 101, 62, 176, 152, 228, 36]

# Split at the unique word
image = []
last_uw = 0
for idx in range(len(bits) - len(unique_word) + 1):
        if bits[idx : idx + len(unique_word)] == unique_word:
            image.append(bits[last_uw : idx + len(unique_word)])
            last_uw = idx + len(unique_word) + 1

# for row in image:
#     print(len(row))

# image = np.array([[0] * n_rows for _ in range(n_cols)])
# for row in range(n_rows):
#     for col in range(n_cols):
#         image[col][row] = slice_LUT(I.pop(0), Q.pop(0))

# if unique_word in image:
#     print('found')

with open(f'data/{input_file}_output.txt', 'w') as outfile:
    for row in image:
        outfile.write(str(row) + '\n')
# image = np.reshape(np.array(data), [n_cols, n_rows])
image = np.array(image).transpose()

if len(image) < 3: # Arbitrary number for error checking, should have more than this many sights of the UW
    print(f"Didn't find enough UWs")
    exit(1)

plt.figure()
plt.imshow(255-image,cmap=plt.get_cmap('Greys'))
plt.title(f'{input_file} with UW')
plt.savefig(f'images/{input_file}_image.png', format='png')

uw_remove = image[:-len(unique_word)]

plt.figure()
plt.imshow(255-uw_remove,cmap=plt.get_cmap('Greys'))
plt.title(f'{input_file} Removed UW')
plt.savefig(f'images/{input_file}_image_removeuw.png', format='png')


