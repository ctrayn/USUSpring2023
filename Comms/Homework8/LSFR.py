import matplotlib.pyplot as plt

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

    def cyclic_autocorrelate(self, tau):
        self.c = [(-1)**b for b in self.outputs]
        summation = 0
        for i in range(self.count):
            summation += self.c[i] * self.c[(i - tau) % self.count]
        return summation / self.count

if __name__ == '__main__':
    ###############
    # Problem 2B
    ###############
    states = [[1,1,0,0], [0,0,0,0], [0,0,0,1], [1,0,1,1]]
    for state in states:
        print(f"\n{state}")
        LFSR(regs=state, xor=[1,1,0,0], count = 8).run()

    ###############
    # Problem 2C
    ###############
    # lfsr = LFSR()
    # lfsr.run()
    # r_c_tau = []
    # spacing = list(range(-lfsr.count, lfsr.count))
    # for tau in spacing:
    #     r_c_tau.append(lfsr.cyclic_autocorrelate(tau))

    # plt.figure()
    # plt.stem(spacing, r_c_tau)
    # plt.title("r_c(tau)")
    # plt.savefig("r_c_tau.png", format='png')

    

