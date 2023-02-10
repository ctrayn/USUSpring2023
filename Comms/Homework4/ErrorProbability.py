from math import sqrt
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt

ERR_COUNT_TARGET = 1e5
# Group of LUTs
Eb = 1

class SignalSpace:
    Es = 0
    A = 0
    std_dev = 0
    LUT = []
    bits_per_symbol = 0

    def __init__(self) -> None:
        pass

    def set_symbol_energy(self):
        print("NO GOOD! WASN'T OVERWRITTEN")

    def random_noise(self, std_dev):
        return np.random.normal(loc=0, scale=std_dev)

    def serial_to_parallel(self, bits:list):
        print("NO GOOD! WASN'T OVERWRITTEN")

    def bits_to_sym(self, bits:list):
        """bits is the parallelized version of the bits (as integers)"""
        print("NO GOOD! WASN'T OVERWRITTEN")

    def simulate_single_symbol(self, std_dev, bits):
        """std_dev is the calculated standard deviation and bits is one symbol worth of bits, returns number of bit errors and symbol errors"""
        print("NO GOOD! WASN'T OVERWRITTEN")

    def get_one_symbol(self, bits:list):
        """Returns the bits list and the correct amount of bits for the given signal space"""
        print("NO GOOD! WASN'T OVERWRITTEN")

    def slice(self, symbol):
        """Turns a single symbol into the closest bits"""
        print("NO GOOD! WASN'T OVERWRITTEN")
    
    def set_LUT():
        """Adjusts the LUT for the given amplitudes"""
        print("NO GOOD! WASN'T OVERRITTEN")

class FourPAM(SignalSpace):
    def __init__(self) -> None:
        super().__init__()
        self.set_symbol_energy()
        self.bits_per_symbol = 2

    def set_symbol_energy(self):
        self.Es = 2 * Eb
        self.A = sqrt(self.Es / 5)
        self.set_LUT()

    def set_LUT(self):
        self.LUT = [-3*self.A, -1*self.A, 1*self.A, 3*self.A]

    def serial_to_parallel(self, bits: list):
        parallel = []
        for idx in range(0,bits,2):
            parallel.append(bits[idx] << 1 | bits[idx + 1])
        return parallel

    def bits_to_sym(self, bits: list):
        assert len(bits) == 2, f"FourPam bit length not 2, received {len(bits)} bits"
        val = bits[0] << 1 | bits[1]
        return self.LUT[val]

    def simulate_single_symbol(self, std_dev, bits):
        sym = self.bits_to_sym(bits)
        sym += self.random_noise(std_dev)
        bits_hat = self.slice(sym)
        bit_errs = 0
        for idx in range(len(bits_hat)):
            if bits_hat[idx] != bits[idx]:
                bit_errs += 1

        sym_err = 0 if bit_errs == 0 else 1
        return bit_errs, sym_err

    def get_one_symbol(self, bits: list):
        return bits[2:], bits[:2]
        
    def slice(self, symbol):
        distances = []
        for look in self.LUT:
            distances.append(abs(symbol-look))
        sym_hat = distances.index(min(distances))
        return [int(i) for i in list(f'{sym_hat:02b}')] # convert to a list of bits

class BPSK(SignalSpace):
    def __init__(self) -> None:
        super().__init__()
        self.set_symbol_energy()
        self.bits_per_symbol = 1

    def set_LUT(self):
        self.LUT = [-1*self.A, 1*self.A]

    def set_symbol_energy(self):
        self.Es = Eb
        self.A = sqrt(self.Es)
        self.set_LUT()

    def serial_to_parallel(self, bits: list):
        return bits

    def bits_to_sym(self, bits: list):
        assert len(bits) == 1, f"BPSK can only take len 1 bits at a time, received len {len(bits)}"
        return self.LUT[bits[0]]

    def get_one_symbol(self, bits: list):
        return bits[1:], bits[:1]

    def slice(self, symbol):
        distances = []
        for look in self.LUT:
            distances.append(abs(symbol - look))
        sym_hat = distances.index(min(distances))
        return [int(sym_hat)]

    def simulate_single_symbol(self, std_dev, bits):
        sym = self.bits_to_sym(bits)
        sym += super().random_noise(std_dev)
        bits_hat = self.slice(sym)
        bit_err = 0
        sym_err = 0
        if bits_hat[0] != bits[0]:
            bit_err = 1
            sym_err = 1
        return bit_err, sym_err

for signal_space in [BPSK(), FourPAM()]:
    bit_errs = []
    for SNRdB in range(0,20+1):
        SNR = 10**(SNRdB/10)
        N0 = SNR/Eb
        sigma = sqrt(N0/1)
        bit_err_count = 0
        sym_err_count = 0
        sym_count = 0
        while bit_err_count < ERR_COUNT_TARGET:
            bits = (rand(signal_space.bits_per_symbol)>0.5).astype(int)
            bit_err, sym_err = signal_space.simulate_single_symbol(sigma, bits)
            bit_err_count += bit_err
            sym_err_count += sym_err
            sym_count += 1
        bit_errs.append(bit_err/(sym_count * signal_space.bits_per_symbol))
        print(sym_count)
    plt.plot(list(range(0, 20+1)), bit_errs)
plt.grid(which='both', axis='both')
plt.yscale('log')
plt.xlabel("Eb/No [dB]")
plt.xlim(0, 20)
plt.ylabel("Pb")
plt.show()