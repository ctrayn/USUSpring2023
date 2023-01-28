from math import sqrt, pi, sin, cos, floor

def srrc1(alpha,N,Lp,Ts):
    """
    Return a vector of the srrc function
    alpha = excess bandwidth
    N = samples per symbol
    Lp = SRRC truncation length #Currently only supports even numbers
    Ts = sample time
    """

    times = []
    number_of_samples = int(floor(Lp/2)) # and then reflect it on the axis?
    for idx in range(number_of_samples):
        t = idx * Ts / N
        times.append(t)
        times.append(-t)
    times.sort()

    # print(len(times))
    # print(times)

    answer = []
    for t in times:
        answer.append(p_of_nT(Ts, alpha, t))

    # print(answer)
    return answer


def p_of_nT(Ts, alpha, t):
    undefined_t_vals = [0, Ts / (4 * alpha)]
    if t in undefined_t_vals:
        return lhopital(Ts, alpha, t)
    else:
        return (1/sqrt(Ts)) * ((sin(pi*(1 - alpha) * t / Ts) + (4 * alpha * t / Ts) * cos(pi * (1 + alpha) * t / Ts))/(((pi*t)/Ts)*(1 - (4 * alpha * t / Ts)**2)))

def lhopital(Ts, alpha, t):
    numerator = (pi * (1 - alpha) / Ts) * cos(pi * (1 - alpha) * t / Ts) + (4 * alpha / Ts) * (cos(pi * (1 + alpha) * t / Ts) - (pi * (1 + alpha) * t / Ts) * sin(pi * (1 + alpha) * t / Ts))
    denominator = pi / sqrt(Ts) - (32 * pi * t / (Ts * sqrt(Ts)))
    return numerator / denominator

def NRZ(Ts):
    return [Ts]

def MANCH(Ts, Lp=60):
    return [-Ts, Ts] + [0] * Lp