from math import sqrt, pi, sin, cos

def srrc1():
    return []


def p_of_nT(N, alpha, n):
    return (1/sqrt(N)) * ((sin(pi*(1 - alpha) * n / N) + (4 * alpha * n / N) * cos(pi * (1 + alpha) * n * N))/(((pi*n)/N)*(1 - (4 * alpha * n / N)**2)))