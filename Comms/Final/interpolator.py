from numpy import sin, pi, linspace
import matplotlib.pyplot as plt

class Interpolator:
    def __init__(self) -> None:
        self.samples = [0] * 4
        self.interpolated_val = 0
        self.color = 'blue'
        self.name = "None"
        self.list_of_vals = []
        self.mu = 0

    def interpolate(self, mu=None):
        """If no mu is given, mu coefficients will not be updated"""
        assert False, "Didn't Overwrite function"

    def new_sample(self, sample):
        #Shift the samples down and insert the new sample
        for index in reversed(range(len(self.samples))):
            if index == 0:
                self.samples[0] = sample
            else:
                self.samples[index] = self.samples[index - 1]

    def calc_mu_coefficients(self, mu):
        assert False, "Didn't Overwrite function"

    def get_result(self):
        return self.interpolated_val

    def store_val(self, val):
        self.list_of_vals.append(val)

    def plot(self,file_name, format='png'):
        plt.figure()
        plt.stem(list(reversed(self.samples)))
        plt.scatter(7 + self.mu, self.interpolated_val)
        plt.show()

class CubicInterpolator(Interpolator):
    def __init__(self) -> None:
        super().__init__()
        self.coefficients = [0] * 4
        self.exponents = [3,2,1,0]
        self.b_coef = [
            [1/6,  0,   -1/6, 0],
            [-1/2, 1/2, 1,    0],
            [1/2,  -1,  -1/2, 1],
            [-1/6, 1/2, -1/3, 0]
        ]
        self.color = 'green'
        self.name = "Cubic"

    def interpolate(self, mu=None):
        if mu != None:
            self.calc_mu_coefficients(mu)
        self.interpolated_val = 0
        for index in range(len(self.samples)):
            self.interpolated_val += self.samples[index] * self.coefficients[index]
        return self.interpolated_val

    def calc_mu_coefficients(self, mu):
        self.mu = mu
        for i in range(len(self.coefficients)):
            self.coefficients[i] = 0
            for l in range(len(self.b_coef)):
                self.coefficients[i] += (mu**self.exponents[l]) * self.b_coef[i][l]
        self.coefficients = list(reversed(self.coefficients))

class LinearInterpolator(Interpolator):
    def __init__(self, num_samples=4) -> None:
        super().__init__()
        self.coefficients = [0]*2
        self.color = 'purple'
        self.name = "Linear"
        self.samples = [0] * num_samples

    def calc_mu_coefficients(self, mu):
        self.mu = mu
        self.coefficients[0] = 1 - mu
        self.coefficients[1] = mu
        # print(f"coef[0] = {self.coefficients[0]} coef[1] = {self.coefficients[1]}")

    def interpolate(self, mu=None):
        if mu != None:
            self.calc_mu_coefficients(mu)
        self.interpolated_val = 0
        print(f"samples x(mk) = {self.samples[1]}, x(mk+1) = {self.samples[2]}")
        print(f"mu = {mu}")
        half_way_index = int(len(self.samples) / 2) - 1
        for index in range(len(self.coefficients)):
            self.interpolated_val += self.samples[half_way_index + index] * self.coefficients[index]
        return self.interpolated_val


if __name__ == '__main__':
    F0 = 1
    T = 0.15 / F0
    spacing = linspace(0,0.5,1000)
    signal = [sin(2*pi*F0*t) for t in spacing]

    mus = linspace(0, 1, 13)
    mu_spacing = [(mu * T) + T for mu in mus]

    sample_times = [3*T, 2*T, T, 0]
    interpolators = [CubicInterpolator(), LinearInterpolator()]
    for interpolator in interpolators:
        for mu in mus:
            for t in sample_times:
                interpolator.new_sample(sin(2*pi*t))
            # TODO: This needs to go over 11 points for mu
            val = interpolator.interpolate(mu)
            interpolator.store_val(val)

    plt.figure()
    plt.plot(spacing, signal)
    plt.title("Interpolators")
    plt.stem(list(reversed(sample_times)), interpolators[0].samples)
    for interp in interpolators:
        plt.plot(mu_spacing, interp.list_of_vals, markerfacecolor=interp.color, markeredgecolor=interp.color)
    plt.legend(["Signal"] + [interp.name for interp in interpolators])
    plt.savefig("interpolators.png", format='png')