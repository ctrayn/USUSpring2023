import numpy as np
import math
import matplotlib.pyplot as plt

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

B = 8
bits2index = 2**np.arange(B-1,-1,-1)
M = 2 ** B # Number of symbols in the constellation
Mroot = math.floor(2**(B/2))
a = np.reshape(np.arange(-Mroot+1,Mroot,2),(2*B,1))
b = np.ones((Mroot,1))
LUT = np.hstack((np.kron(a,b), np.kron(b,a)))
Enorm = np.sum(LUT ** 2) / M
LUT = LUT/math.sqrt(Enorm)

# plt.figure(1)
# # Plot the constellation
# plt.plot(LUT[:,0],LUT[:,1],'o');
# for i in range(0,M):
#     plt.text(LUT[i,0]+0.02,LUT[i,1]+.02,i)
# # grid on; axis((max(axis)+0.1/B)*[-1 1 -1 1]); axis square;
# plt.xlabel('In-Phase')
# plt.ylabel('Quadrature')
# plt.title('Constellation Diagram');
# plt.show()

def slice_LUT(I, Q):
    distances = []
    for index in range(len(LUT)):
        distances.append(math.sqrt((LUT[index][0] - I)**2 + (LUT[index][1] - Q)**2))
    min_index = distances.index(min(distances))
    return min_index

def slice_LUT_get_symbols(I, Q):
    value = slice_LUT(I, Q)
    a1 = value & 0xFFFF
    a0 = (value >> 32) & 0xFFFF
    return LUT[a0][1], LUT[a1][1]
