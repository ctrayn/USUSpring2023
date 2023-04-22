from math import sqrt, pi, cos, sin, atan2

class PLL:
    def __init__(self) -> None:
        self.BnT = 0.5
        self.Z = 1 / sqrt(2)
        self.Kp = 1
        self.K0 = 5.66
        self.compute_K1_K2(self.BnT, self.Z, self.K0, self.Kp)
        self.Omega_n = 1e9
        self.A = 1
        self.delayed_k2 = 0
        self.delayed_dds = 0
        self.theta_hat = 130
        self.error_signal = []

    def compute_K1_K2(self, BnT, Z, K0, Kp):
        K0_Kp_K1 = (4 * Z * BnT / (Z + 1 / (4 * Z)))  / (1 + 2 * Z * BnT / (Z + 1 / (4 * Z)) + (BnT / (Z + 1 / (4 * Z)))**2)
        K0_Kp_K2 = (4 * (BnT / (Z + 1 / (4 * Z)))**2) / (1 + 2 * Z * BnT / (Z + 1 / (4 * Z)) + (BnT / (Z + 1 / (4 * Z)))**2)

        self.K1 = K0_Kp_K1 / (K0 * Kp)
        self.K2 = K0_Kp_K2 / (K0 * Kp)

    def pll(self, x, y, a0, a1):
        theta_error = cos(atan2(x, y)) * -sin(atan2(a0, a1))
        self.error_signal.append(theta_error)
        return theta_error

        # Loop Filter
        self.delayed_k2 = (theta_error * self.K2) + self.delayed_k2
        v = self.delayed_k2 + (theta_error * self.K1)

        # DDS
        self.theta_hat = self.delayed_dds
        self.delayed_dds = -self.delayed_dds + (self.K0 * v)
        return cos(self.theta_hat) + (pi/4)