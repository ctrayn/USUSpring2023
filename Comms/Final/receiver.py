import math
import numpy as np
import matplotlib.pyplot as plt
from tx_rx import *
from pulses import srrc1, slice_LUT

input_file = 'test_2023'

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
    sample_time=N,
    Lp=Lp,
    pulse=pulse,
    diff_filter_len=diff_filter_len,
    diff_T=diff_T,
    Omega0=Omega0)
I, Q = rx.get_sampled_signal()

plt.figure()
plt.scatter(I, Q)
plt.savefig(F"images/{input_file}_IQ.png", format='png')

##############################
# Bits to Image
##############################

n_rows = slice_LUT(I.pop(0), Q.pop(0))
n_cols = slice_LUT(I.pop(0), Q.pop(0))

bits = []
for bit in range(len(I)):
    bits.append(slice_LUT(I.pop(0), Q.pop(0)))

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

# FIXME: Remove the first 23 points
for i in range(23):
    image[0].pop(0)

for row in image:
    print(len(row))

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

plt.figure()
plt.imshow(255-image,cmap=plt.get_cmap('Greys'))
plt.savefig(f'images/{input_file}_image.png', format='png')


