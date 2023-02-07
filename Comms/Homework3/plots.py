import matplotlib.pyplot as plt
import numpy as np
import time
import sys

def eye_diagram(sig:np.array, Lp, N, name=''):
    offset = (2*Lp - np.floor(N/2)).astype(int)
    Nsymtoss = 2*np.ceil(Lp/N) # throw away symbols at the end
    nc = (np.floor((len(sig) - offset - Nsymtoss*N)/N)).astype(int) # number of points of signal to plot
    xreshape = sig[offset:offset + nc*N].reshape(nc,N)
    plt.figure(5); plt.clf()
    plt.plot(np.arange(-np.floor(N/2), np.floor(N/2)), xreshape.T, color='b', linewidth=0.25)
    plt.title(f"Eye Diagram {name}")
    plt.savefig(f"images/Eye_Diagram_{name}.png")
    # plt.show()
    plt.close()

def power_spectra(sig, name=''):
    plt.figure()
    plt.plot(10*np.log10(np.abs(np.fft.fftshift(np.fft.fft(np.correlate(sig,sig,'full'))))**2))
    plt.title(f"Power Spectra {name}")
    plt.savefig(f"images/Power_Spectra_{name}.png")
    plt.close()

def plot_and_show(y, x:list=[], title:str=None, xlabel=None, ylabel=None, grid=None, plot=None, axis=False):
    plt.figure()

    if not plot:
        plot_type = plt.plot
    elif plot == 'stem':
        plot_type = plt.stem
    elif plot == 'scatter':
        plot_type = plt.scatter
    else:
        print(f"Unkown plot type {plot}")
        sys.exit(1)

    if x == []:
        plot_type(y)
    else:
        plot_type(y,x)

    if title:
        plt.title(title)

    if xlabel:
        plt.xlabel(xlabel)

    if ylabel:
        plt.ylabel(ylabel)

    if grid == 'log':
        plt.grid(which='both')
        plt.scale('log')
    elif grid == True:
        plt.grid()

    if axis:
        plt.axhline(0, color='black', linewidth=.5)
        plt.axvline(0, color='black', linewidth=.5)

    # plt.show()
    if not title:
        plt.savefig(f"images/{time.time()}.png")
    else:
        plt.savefig(f"images/{title}.png")

    plt.close()