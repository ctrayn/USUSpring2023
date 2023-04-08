import matplotlib.pyplot as plt
from differentiate import Differentiator
from interpolator import *
from ted import TED
from pulses import *
from tx_rx import *

K = 1
N = 160
Lp = 160
num_bits = 100
Ts = 1
alpha = 1
pulse = srrc1(alpha, N, Lp, Ts)

########################
# Transmit
########################

tx = TX(num_bits, pulse, N)
tx.plot_signal()
I_sent, Q_sent = tx.get_signal()
sent_bits = tx.bits


########################
# Receive
########################

diff_filter_len = 11
diff_T = 1

rx = RX(I_sent, Q_sent, 10, Lp, pulse, diff_filter_len, diff_T)
rx.plot_sampled_signal() # This has been through the matched filter and derivitive filter

rx_I, rx_Q = rx.get_sampled_signal()
rx_Ip, rx_Qp = rx.get_sampled_deriv()

assert len(rx_I) == len(rx_Q) == len(rx_Ip) == len(rx_Qp), "RX Signal and Derivitive are not the same length"

plt.figure()
plt.scatter(rx_I, rx_Q)
plt.title("Received IQ")
plt.savefig("rx_IQ.png", format='png')

########################
# Interpolation
########################

num_samples = 16
ted = TED(num_samples=num_samples)
I_int = LinearInterpolator(num_samples=num_samples)
Q_int = LinearInterpolator(num_samples=num_samples)
Ip_int = LinearInterpolator(num_samples=num_samples)
Qp_int = LinearInterpolator(num_samples=num_samples)

I_results = [0]
Q_results = [0]
Ip_results = [0]
Qp_results = [0]
mus = []
for index in range(len(rx_I)):
    # New samples
    I_int.new_sample(rx_I[index])
    Q_int.new_sample(rx_Q[index])
    Ip_int.new_sample(rx_Ip[index])
    Qp_int.new_sample(rx_Qp[index])

    # Give the ted the last signal and derivitie samples
    mu = ted.timing_error(I_results[-1], Q_results[-1], Ip_results[-1], Qp_results[-1])
    mus.append(mu)
    if ted.strobe:
        I_int.interpolate(mu); Q_int.interpolate(mu); Ip_int.interpolate(mu); Qp_int.interpolate(mu)
        # I_int.plot("I_int")
        I_results.append(I_int.get_result())
        Q_results.append(Q_int.get_result())
        Ip_results.append(Ip_int.get_result())
        Qp_results.append(Qp_int.get_result())
I_results.pop(0)
Q_results.pop(0)
Ip_results.pop(0)
Qp_results.pop(0)

recieved_bits = []
for I, Q in zip(I_results, Q_results):
    a0, a1 = rx.slice_QPSK(I, Q)
    recieved_bits.append(a0)
    recieved_bits.append(a1)
print(f"Do the rx and tx bits match? {'Yes' if recieved_bits == sent_bits else 'No'} ")

plt.figure()
plt.plot(mus)
plt.title("Mu(k)")
plt.savefig(f"mus_{I_int.name}.png", format='png')

plt.figure()
plt.plot(ted.es)
plt.title("Filter Error")
plt.savefig(f"e_{I_int.name}.png", format='png')

plt.figure()
plt.scatter(I_results, Q_results)
plt.title("Recovered IQ")
plt.savefig(f"recovered_IQ_{I_int.name}.png", format='png')
