from numpy import linspace, pi, sin, cos
import matplotlib.pyplot as plt

class Differentiator:
    def __init__(self, T, Wc, filter_len = 11) -> None:
        self.filter_len = filter_len
        self.L = int((self.filter_len - 1) / 2)
        self.T = T
        self.Wc = Wc
        self.inputs = [0] * filter_len
        self.indeces = list(range(-self.L, self.L + 1, 1))
        self.filter = []
        self.output = []
        self.make_filter()
        assert len(self.filter) == self.filter_len, f'Expected: {self.filter_len}, Actual {len(self.filter)}'

    def make_filter(self):
        for n in self.indeces:
            if n == 0:
                self.filter.append(0)
            else:
                self.filter.append((self.Wc * self.T / (pi * self.T) * cos(self.Wc * self.T * n) /  n) - (1/(pi*self.T) * sin(self.Wc*self.T*n) / (n**2)))
    
    def plot_filter(self):
        plt.figure()
        plt.stem(self.indeces, self.filter)
        plt.savefig(f"filter_{self.T}.png", format='png')

    def get_result(self):
        return self.output

    def update_input(self, input):
        self.inputs.insert(0, input)
        self.inputs.pop(-1)
            
    def differentiate(self, input):
        self.update_input(input)
        summation = 0
        for n in range(len(self.inputs)):
            summation += self.filter[n] * self.inputs[n]
        self.output.append(summation)

if __name__ == '__main__':
    for Wc in [10, 1000, 2000, 5000, 10000]:
        T = pi / Wc
        num_points = 1000
        spacing = linspace(0, 2*pi, num_points)
        A = 1
        signal = [A * sin(Wc*t) for t in spacing]
        diff = Differentiator(T, Wc, filter_len=17)
        diff.plot_filter()

        for sample in signal:
            diff.differentiate(sample)

        result = diff.get_result()

        plt.figure()
        plt.title(f"Differentiator for {Wc}")
        plt.plot(spacing, signal)
        plt.plot(spacing, result)
        plt.legend(["Sin Input", "Differentiator Ouput"])
        plt.savefig(f"diff_{Wc}.png", format='png')

        error = []
        for i in range(diff.filter_len + 1, len(signal)):
            ans = A * Wc * cos(Wc * spacing[i])
            error.append(ans - result[i])
        plt.figure()
        plt.plot(spacing[diff.filter_len + 1:], error)
        plt.savefig(f"error_{Wc}.png", format='png')

    

