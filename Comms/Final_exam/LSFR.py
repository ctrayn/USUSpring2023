import matplotlib.pyplot as plt
from math import sqrt
from numpy import linspace

class LFSR:
    def __init__(self, regs=[1,0,0,0], xor = [1,0,0,0], count=16) -> None:
        self.regs = regs
        self.xor = xor
        self.output = self.regs[-1]
        self.count = count
        self.outputs = [self.output]
        self.c = []

    def run(self):
        self.print_header()
        for i in range(self.count):
            self.print_state(i)
            for index in reversed(range(len(self.regs))):
                if index == 0:
                    self.regs[0] = self.output
                else:
                    if self.xor[index - 1]:
                        self.regs[index] = self.regs[index - 1] ^ self.output
                    else:
                        self.regs[index] = self.regs[index - 1]
            self.output = self.regs[-1]
            self.outputs.append(self.output)
        self.outputs.pop(-1)

    def print_header(self):
        print("| Count |  State  | Output |")
        print("|-------|---------|--------|")

    def print_state(self, count):
        print(f"| {count:^5} | {self.regs[0]} {self.regs[1]} {self.regs[2]} {self.regs[3]} | {self.output:^6} |")

if __name__ == '__main__':
    lfsr = LFSR(regs=[1,0,0,0,0], xor=[0,1,0,0,0], count=35)
    lfsr.run()
    ct = [(-1)**b for b in lfsr.outputs]

    Tc = 0.25
    num_points_in_pulse = 10

    signal = []
    for c in ct:
        for num in range(num_points_in_pulse):
            signal.append(1/sqrt(Tc) * c)

    spacing = linspace(0, Tc * len(ct), num_points_in_pulse * len(ct))
    
    plt.figure()
    plt.plot(spacing, signal)
    plt.title(f"c(t) Tc={Tc}")
    plt.xlabel("t")
    plt.ylabel('c(t)')
    plt.savefig('images/c_t_lfsr.png', format='png')

    Ts = 5*Tc
    bits = [0, 1, 0, 1, 1, 0, 1]
    bits_sym = [-1 if b == 0 else 1 for b in bits]
    bpsk_signal = []
    for b in bits_sym:
        for num in range(5 * num_points_in_pulse):
            bpsk_signal.append(b / sqrt(Ts))

    bpsk_spacing = linspace(0, Ts * len(bits_sym), num_points_in_pulse * 5 * len(bits_sym))
    plt.figure()
    plt.plot(bpsk_spacing, bpsk_signal)
    plt.title(f's(t) Ts={Ts}')
    plt.xlabel("t")
    plt.ylabel('s(t)')
    plt.savefig('images/s_t_lfsr.png', format='png')

    xt = [ct[idx % len(ct)] * bpsk_signal[idx] for idx in range(len(bpsk_signal))]
    plt.figure()
    plt.plot(bpsk_spacing, xt)
    plt.title("x(t)")
    plt.xlabel("t")
    plt.ylabel("x(t)")
    plt.savefig('images/x_t_lfsr.png', format='png')

