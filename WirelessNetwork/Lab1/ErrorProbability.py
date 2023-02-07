from math import sqrt

# Group of LUTs
Eb = 1

for SNRdB in range(1:20+1):
    SNR = 10**(SNRdB/10)
    N0 = SNR/Eb
    sigma = sqrt(N0/1)
    bit_err_count = 0
    sym_err_count = 0
    while bit_err_count < ERR_COUNT_TARGET: