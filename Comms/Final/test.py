import matplotlib.pyplot as plt
import numpy as np
import math

#########################
# Transmit Pulse
#########################

B = 8; # Bits per symbol (B should be even: 8, 6, 4, 2)
# B = 4;
bits2index = 2**np.arange(B-1,-1,-1)
M = 2 ** B # Number of symbols in the constellation
Mroot = math.floor(2**(B/2))
a = np.reshape(np.arange(-Mroot+1,Mroot,2),(2*B,1))
b = np.ones((Mroot,1))
LUT = np.hstack((np.kron(a,b), np.kron(b,a)))
# will be of the form (for example)
# -3, -3
# -3, -1
# -3, 1
# ...
# 3, 3
# of shape (B^2, 2)
# Scale the constellation to have unit energy
Enorm = np.sum(LUT ** 2) / M
LUT = LUT/math.sqrt(Enorm)

print(LUT)

plt.figure()
plt.plot(srrc)
plt.savefig("images/LUT.png", format='png')