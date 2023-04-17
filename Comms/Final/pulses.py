import numpy as np
import math

def srrc1(alpha, N, Lp):
    """
    Return a vector of the srrc function
    alpha = excess bandwidth
    N = samples per symbol
    Lp = SRRC truncation length #Currently only supports even numbers
    """

    EB = alpha

    t = np.arange(-Lp*N,Lp*N+1) /N + 1e-8; # +1e-8 to avoid divide by zero
    tt = t
    srrc = ((np.sin(math.pi*(1-EB)*tt)+ 4*EB*tt * np.cos(math.pi*(1+EB)*tt))/((math.pi*tt)*(1-(4*EB*tt)**2)))
    srrc = srrc/math.sqrt(N)
    return srrc

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

def slice_LUT(I, Q):
    distances = []
    for index in range(len(LUT)):
        distances.append(math.sqrt((LUT[index][0] - I)**2 + (LUT[index][1] - Q)**2))
    min_index = distances.index(min(distances))
    bits = f"{min_index:08b}"
    return bits
