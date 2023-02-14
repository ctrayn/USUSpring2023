from math import sqrt, cos, sin, floor, ceil
import numpy as np
from numpy.random import rand
import matplotlib.pyplot as plt
import copy

class SignalSpace:
    Es = 0
    Eb = 0
    A = 0
    std_dev = 0
    LUT = []
    bits_per_symbol = 0
    num_axis = 0
    name = "SignalSpace"
    bit_errs = []
    sym_errs = []

    def __init__(self) -> None:
        pass

    def set_symbol_energy(self):
        print("NO GOOD! WASN'T OVERWRITTEN")

    def random_noise(self, std_dev):
        return np.random.normal(loc=0, scale=std_dev)

    def bits_to_amp(self, bits:list):
        """bits is the parallelized version of the bits (as integers)"""
        print("NO GOOD! WASN'T OVERWRITTEN")

    def simulate_single_symbol(self, std_dev, bits):
        """std_dev is the calculated standard deviation and bits is one symbol worth of bits, returns number of bit errors and symbol errors"""
        assert len(bits) == self.bits_per_symbol
        sym = self.bits_to_amp(bits)
        sym = [amp + self.random_noise(std_dev) for amp in sym] # Add noise
        bits_hat = self.slice(sym)
        bit_err = 0
        sym_err = 0
        for idx in range(len(bits)):
            if bits_hat[idx] != bits[idx]:
                bit_err += 1
                sym_err = 1 # we want to max out at 1 per call to the function
        return bit_err, sym_err

    def slice(self, symbol):
        """Turns a single symbol into the closest bits"""
        print("NO GOOD! WASN'T OVERWRITTEN")
    
    def set_LUT(self):
        """Adjusts the LUT for the given amplitudes"""
        print("NO GOOD! WASN'T OVERRITTEN")

    def plot(self):
        """Plot and save the data"""
        plt.figure()
        plt.clf()
        plt.plot(self.bit_errs)
        plt.plot(self.sym_errs)
        plt.grid(which='both', axis='both')
        plt.yscale('log')
        plt.xlabel("Eb/No [dB]")
        plt.xlim(0, 15)
        plt.ylabel("Pb")
        plt.ylim(1e-6, 0)
        plt.title(f"Probability of error for {self.name}")
        plt.legend(["Bit errors", "Symbol errors"])
        plt.savefig(f"{self.name}.png")

    def simulate(self, dBRangeStart=0, dBRangeEnd=15, Eb=1, ErrCountTarget=1e2, ErrStopLimit=1e-6):
        self.bit_errs = []
        self.sym_errs = []
        for SNRdB in range(dBRangeStart,dBRangeEnd+1):
            SNR = 10**(SNRdB/10)
            N0 = Eb/SNR
            sigma = sqrt(N0/2)
            # print(sigma)
            bit_err_count = 0
            sym_err_count = 0
            sym_count = 0
            while bit_err_count < ErrCountTarget:
                # print(bit_err_count)
                bits = (rand(self.bits_per_symbol)>0.5).astype(int)
                # print(bits)
                bit_err, sym_err = self.simulate_single_symbol(sigma, bits)
                bit_err_count += bit_err
                sym_err_count += sym_err
                sym_count += 1
                if bit_err_count > 1 and (bit_err_count/(sym_count * self.bits_per_symbol)) < ErrStopLimit:
                    break
                print(f"\033[K{self.name:5} SNRdB [{SNRdB}/{dBRangeEnd}] Error Count [{bit_err_count}/{ErrCountTarget}] {bit_err_count/(sym_count * self.bits_per_symbol)}", end='\r')
            self.bit_errs.append(bit_err_count/(sym_count * self.bits_per_symbol))
            self.sym_errs.append(sym_err_count/sym_count)
            # print(f"\n{self.bit_errs[-1]}")

            if self.bit_errs[-1] < ErrStopLimit:
                break
        print("")
        self.plot()
        with open(f"{self.name}_bit.dat", 'w') as w:
            for item in self.bit_errs:
                w.write(f"{item}\n")
        with open(f"{self.name}_sym.dat", 'w') as w:
            for item in self.sym_errs:
                w.write(f"{item}\n")
        

class BPSK(SignalSpace):
    def __init__(self, Eb=1) -> None:
        super().__init__()
        self.name = "BPSK"
        self.bits_per_symbol = 1
        self.num_axis = 1
        self.Eb = Eb
        self.set_symbol_energy()

    def set_LUT(self):
        self.LUT = [[-self.A], [self.A]]
        # print(f"LUT: {self.LUT} A:{self.A}")

    def set_symbol_energy(self):
        self.Es = self.Eb
        self.A = sqrt(self.Es)
        self.set_LUT()

    def bits_to_amp(self, bits: list):
        assert len(bits) == 1, f"BPSK can only take len 1 bits at a time, received len {len(bits)}"
        return copy.deepcopy(self.LUT[bits[0]])

    def slice(self, symbol):
        if symbol[0] >= 0:
            return [1]
        else:
            return [0]

class QPSK(SignalSpace):
    def __init__(self, Eb=1) -> None:
        super().__init__()
        self.bits_per_symbol = 2
        self.name = "QPSK"
        self.num_axis = 2
        self.Eb = Eb
        self.set_symbol_energy()

    def set_LUT(self):
        self.LUT = [[self.A, self.A], [self.A, -self.A], [-self.A, self.A], [-self.A, -self.A]]

    def set_symbol_energy(self):
        self.Es = self.Eb *1.25
        self.A = self.Es / sqrt(2)
        self.set_LUT()

    def bits_to_amp(self, bits: list):
        assert len(bits) == self.bits_per_symbol
        val = bits[0] << 1 | bits[1]
        return copy.deepcopy(self.LUT[val])

    def slice(self, symbol):
        assert len(symbol) == self.bits_per_symbol
        bits = []
        if symbol[0] >= 0:
            bits.append(0)
        else:
            bits.append(1)
        
        if symbol[1] >= 0:
            bits.append(0)
        else:
            bits.append(1)

        return bits

class EightPSK(SignalSpace):
    def __init__(self, Eb=1) -> None:
        super().__init__()
        self.bits_per_symbol = 3
        self.name = "8PSK"
        self.num_axis = 2
        self.Eb = Eb
        self.set_symbol_energy()

    def set_LUT(self):
        A = self.A
        for theta in [5*np.pi/4, np.pi, np.pi/2, 3*np.pi/4, 3*np.pi/2, 7*np.pi/4, np.pi/4, 0]:
            self.LUT.append([A * cos(theta), A * sin(theta)])

    def set_symbol_energy(self):
        self.Es = 3 * self.Eb
        self.A = (-1 + sqrt(1 - 4 * (-6 * self.Eb))) / 2
        self.set_LUT()

    def bits_to_amp(self, bits: list):
        assert len(bits) == self.bits_per_symbol
        val = (bits[0] << 2) | (bits[1] << 1) | (bits[2] << 0)
        return copy.deepcopy(self.LUT[val])

    def slice(self, symbol):
        distances = []
        for look in self.LUT:
            distances.append(sqrt((symbol[0] - look[0])**2) + (symbol[1] - look[1])**2)
        sym_hat = distances.index(min(distances))
        return [(sym_hat >> 2) & 1, (sym_hat >> 1) & 1, sym_hat & 1]

class CCITT(SignalSpace):
    def __init__(self, Eb=1) -> None:
        super().__init__()
        self.name = "CCITT"
        self.bits_per_symbol = 3
        self.num_axis = 2
        self.Eb = Eb
        self.set_symbol_energy()

    def set_LUT(self):
        A = self.A
        self.LUT = [[2*A, 0], [0, 2*A], [-2*A, 0], [0, -2*A], [A, A], [-A, A], [-A, -A], [A, -A]]
    
    def set_symbol_energy(self):
        self.Es = 3 * self.Eb
        self.A = self.Es / (1 + sqrt(2))

    def bits_to_amp(self, bits: list):
        assert len(bits) == self.bits_per_symbol
        val = (bits[0] << 2) | (bits[1] << 1) | (bits[2])
        return copy.deepcopy(self.LUT[val])

    def slice(self, symbol):
        distances = []
        for look in self.LUT:
            distances.append(sqrt((symbol[0] - look[0])**2 + (symbol[1] - look[1])**2))
        sym_hat = distances.index(min(distances))
        return [(sym_hat >> 2) & 1, (sym_hat >> 1) & 1, sym_hat & 1]

# signal_spaces = [QPSK()]
signal_spaces = [BPSK(), QPSK(), EightPSK(), CCITT()]

for signal_space in signal_spaces:
    signal_space.simulate(dBRangeStart=0, dBRangeEnd=15, ErrCountTarget=1e2, ErrStopLimit=1e-6)

plt.figure()
plt.clf()
for space in signal_spaces:
    plt.plot(space.bit_errs)
plt.grid(which='both', axis='both')
plt.yscale('log')
plt.xlabel("Eb/N0 [dB]")
plt.xlim(0,15)
plt.tight_layout()
plt.ylabel("Pb")
plt.ylim(1e-6, 0)
plt.title("Probability of Error")
plt.legend([space.name for space in signal_spaces])
plt.savefig('joined.png')