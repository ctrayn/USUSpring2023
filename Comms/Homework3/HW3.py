import numpy as np
from numpy.random import rand
from pulses import *
from plots import *
from tx import *
from rx import *

def transmit_and_receive(bits, Nsym, LUT, bits_per_sym, pulse, Lp, Ts, omega, keyword=''):

    parallel = serial_to_parallel(bits, bits_per_sym)
    # plot_and_show(parallel, title="Parallel", plot='stem')

    I,Q = parallel_to_I_Q(parallel, LUT)
    # plot_and_show(I, title="I(t)", plot='stem')
    # plot_and_show(Q, title="Q(t)", plot='stem')
    # plot_and_show(Q, I, title="I vs Q")

    I_upsampled = upsample(I, N)
    Q_upsampled = upsample(Q, N)
    # plot_and_show(I_upsampled, title="I(t) upsampled", plot='stem')
    # plot_and_show(Q_upsampled, title="Q(t) upsampled", plot='stem')

    I_shaped = np.convolve(I_upsampled, pulse)
    Q_shaped = np.convolve(Q_upsampled, pulse)
    # plot_and_show(I_shaped, title="I shaped")
    # plot_and_show(Q_shaped, title="Q_shaped")

    ######################################
    #TODO: Send and receive with cos and sin
    ######################################

    Ir = I_shaped
    Qr = Q_shaped

    I_sig = [i * sqrt(2) *  cos(omega) for i in I_shaped]
    Q_sig = [q * sqrt(2) * -sin(omega) for q in Q_shaped]

    st = [I_sig[i] + Q_sig[i] for i in range(len(I_sig))]
    plot_and_show(st, title="Signal")

    # rt = st # Transmit over the channel, noiseless

    # Ir = [i * sqrt(2) *  cos(omega) for i in rt]
    # Qr = [q * sqrt(2) * -sin(omega) for q in rt]

    xt = matched_filter(Ir, pulse)
    yt = matched_filter(Qr, pulse)

    # plot_and_show(xt, title="x(t)")
    # plot_and_show(yt, title="y(t)")

    xt_sampled = sample_signal(xt, sample_time=N, Lp=Lp)
    yt_sampled = sample_signal(yt, sample_time=N, Lp=Lp)

    # plot_and_show(xt_sampled, title="Sampled x(t)", plot='stem')
    # plot_and_show(yt_sampled, title="Sampled y(t)", plot='stem')

    xt_normalized = normalize_amplitude(xt_sampled, max(max(LUT)))
    yt_normalized = normalize_amplitude(yt_sampled, max(max(LUT)))

    # plot_and_show(xt_normalized, title="Noramlized x(t)", plot='stem')
    # plot_and_show(yt_normalized, title="Normazlied y(t)", plot='stem')

    ###################
    # Phase Trajectory
    ####################
    plot_and_show(y=yt_normalized, x=xt_normalized, title="Phase_Trajectory" + keyword, xlabel="x(t)", ylabel="y(t)", axis=True)

    ###################
    # Scatter Plot
    ####################
    plot_and_show(y=yt_normalized, x=xt_normalized, title="Scatter_Plot" + keyword, plot='scatter', xlabel="x(t)", ylabel='y(t)', axis=True)

    ###################
    # Eye diagram
    ###################
    eye_diagram(xt, Lp, N, name=f"Inphase_{keyword}")
    eye_diagram(yt, Lp, N, name=f"Quadurature_{keyword}")

    ##################
    # Power Spectra
    ##################
    power_spectra(Ir, name="Ir_"+keyword)
    power_spectra(st, name="s(t)_"+keyword)
    power_spectra(xt, name="x(t)_"+keyword)

    bits_received = slice(xt_normalized, yt_normalized, LUT)
    return bits_received

# Constants
ALPHA_1=1
ALPHA_HALF=0.5
LUT_2PAM = [-1,1]
LUT_4PAM = [-2,-1,1,2]
LUT_4QAM = [[-1,-1], [-1,1], [1,-1], [1,1]]
LUT_16QAM = [[-3,-3], [-3,-1], [-3,3], [-3,1], [-1,-3], [-1,-1], [-1,3], [-1,1], [1,-3], [1,-1], [1,3], [1,1], [3,-3], [3,-1], [3,3], [3,1]]

# Contant variables
Nsym = 500
N = 1000
Lp = 6000
Ts = 1
omega = 100

for LUT, bits_per_sym in [[LUT_16QAM, 4], [LUT_4QAM, 2]]:
    for alpha in [ALPHA_1, ALPHA_HALF]:
        pulse = srrc1(alpha, N, Lp, Ts)
        if len(LUT) == 1:
            PQ = 'PAM'
        else:
            PQ = 'QAM'
        keyword = f'_{len(LUT)}{PQ}_ALPHA{alpha}'

        bits = (rand(Nsym*(bits_per_sym))> 0.5).astype(int) # generate random bits {0,1}

        bits_received = transmit_and_receive(bits=bits, Nsym=Nsym, LUT=LUT, bits_per_sym=bits_per_sym, pulse=pulse, Lp=Lp, Ts=Ts, omega=omega, keyword=keyword)

        print(f"Do the transmitted bits match the recevied bits? {np.array_equiv(bits, bits_received)}")

# For QPSK
bits = (rand(Nsym*(bits_per_sym))> 0.5).astype(int) # generate random bits {0,1}
N = 100
Lp = 600
Ts=1
omega=100
srrcout = srrc1(alpha=1, N=N, Lp=Lp, Ts=Ts)
bits_received = transmit_and_receive(bits=bits, Nsym=1000, LUT=LUT_4QAM, bits_per_sym=2, pulse=srrcout, Lp=Lp, Ts=Ts, omega=omega, keyword='QPSK')
print(f"Do the transmitted bits match the recevied bits? {np.array_equiv(bits, bits_received)}")
