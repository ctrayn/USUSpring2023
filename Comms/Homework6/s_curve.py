#!/bin/python3
import matplotlib.pyplot as plt
from numpy import linspace, sign
from math import pi, sin, cos
sim_number = 100

def gen_s_curve(constellation_points, filename, K=1):
    data_points = []
    for theta in linspace(-pi, pi, sim_number):
        averages = []
        for point in constellation_points:
            xp = (cos(theta) - sin(theta)) * point[0]
            yp = (sin(theta) + cos(theta)) * point[1]
            a0 = sign(xp)
            a1 = sign(yp)
            ek = (yp * a0) - (xp * a1)
            averages.append(ek)
        data_points.append(sum(averages) / len(averages))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('center')
    ax.spines['top'].set_color('none')
    xtick_points = [-pi,-3*pi/4, -pi/2, -pi/4, pi/4, pi/2, 3*pi/4, pi]
    xtick_labels = ['-π', '', '-π/2', '', '', 'π/2', '', 'π']
    plt.xticks(xtick_points, xtick_labels)
    plt.title(filename.split('.')[0])
    plt.plot(linspace(-pi,pi,sim_number),data_points)
    plt.savefig(filename, format='png')

BPSK = [[-1,0], [1,0]]
eightPSK = [[cos(angle), sin(angle)] for angle in [0, 45, 90, 135, 180, 225, 270, 315, 360]]
sixteenQAM = [
    [ 3,-3], [ 3,-1], [ 3,1], [ 3,3],
    [ 1,-3], [ 1,-1], [ 1,1], [ 1,3],
    [-1,-3], [-1,-1], [-1,1], [-1,3],
    [-3,-3], [-3,-1], [-3,1], [-3,3],
]

gen_s_curve(BPSK, "BPSK.png")
gen_s_curve(eightPSK, "8PSK.png")
gen_s_curve(sixteenQAM,"16QAM.png")
    