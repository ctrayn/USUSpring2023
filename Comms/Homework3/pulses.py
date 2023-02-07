from math import sqrt, pi, sin, cos, floor

def srrc1(alpha, N, Lp, Ts):
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
        times.append(-t)
        times.append(t)
    times.remove(0) # Remove the second zero
    times.sort()

    answer = []
    for t in times:
        answer.append(p_of_nT(Ts, alpha, t))

    while None in answer:
        index = answer.index(None)
        value = (answer[index-1] + answer[index+1])/2
        answer[index] = value

    return answer


def p_of_nT(Ts, alpha, t):
    undefined_t_vals = [0, Ts / (4 * alpha)]
    try:
    # if t in undefined_t_vals:
        # return lhopital(Ts, alpha, t)
    # else:
        return (1/sqrt(Ts)) * ((sin(pi*(1 - alpha) * t / Ts) + (4 * alpha * t / Ts) * cos(pi * (1 + alpha) * t / Ts))/(((pi*t)/Ts)*(1 - (4 * alpha * t / Ts)**2)))
    except ZeroDivisionError:
        return None
        # return lhopital(Ts, alpha, t)

def lhopital(Ts, alpha, t):
    numerator = (pi * (1 - alpha) / Ts) * cos(pi * (1 - alpha) * t / Ts) + (4 * alpha / Ts) * (cos(pi * (1 + alpha) * t / Ts) - (pi * (1 + alpha) * t / Ts) * sin(pi * (1 + alpha) * t / Ts))
    denominator = pi / sqrt(Ts) - (32 * pi * t / (Ts * sqrt(Ts)))
    return numerator / denominator

def NRZ(Ts):
    return [Ts]

def MANCH(Ts, Lp=60):
    return [-Ts, Ts] + [0] * Lp