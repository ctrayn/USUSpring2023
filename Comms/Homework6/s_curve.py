#!/usr/bin/python3
import sys
from math import pi
from numpy import linspace
import matplotlib.pyplot as plt

class SCurve:

    def __init__(self):
        self.name = "Unset"
        self.num_points = 1000
        self.points = []
        self.graph = []

    def get_point(self, point):
        print("Not Overridden")
    
    def gen_curves(self):
        print("Not Overridden")

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.spines['left'].set_position('center')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('center')
        ax.spines['top'].set_color('none')
        xtick_points = [-pi,-3*pi/4, -pi/2, -pi/4, pi/4, pi/2, 3*pi/4, pi]
        xtick_labels = ['-π', '', '-π/2', '', '', 'π/2', '', 'π']
        plt.xticks(xtick_points, xtick_labels)
        ytick_points = [-pi,-3*pi/4, -pi/2, -pi/4, pi/4, pi/2, 3*pi/4, pi]
        ytick_labels = ['-π', '', '-π/2', '', '', 'π/2', '', 'π']
        plt.yticks(ytick_points, ytick_labels)
        plt.title(self.name)


class BPSK_data_aided(SCurve):
    def __init__(self):
        super().__init__()
        self.name = "BPSK Data Aided"
        self.graph = []
        self.points = linspace(-pi, pi, self.num_points)
        self.gen_curves()

    def get_point(self, point):
        return point
    
    def gen_curves(self):
        for point in self.points:
            self.graph.append(self.get_point(point))

    def plot(self):
        super().plot()
        plt.plot(self.points, self.graph)
        plt.savefig(self.name + '.png', format='png')

    
class BPSK_decision_directed(SCurve):
    def __init__(self):
        super().__init__()
        self.name = "BPSK Decision Directed"
        self.graph = [[],[],[]]
        self.gen_curves()

    def get_point(self, point):
        if point < -pi/2:
            self.graph[0].append(point + pi)
        elif point < pi/2:
            self.graph[1].append(point)
        else:
            self.graph[2].append(point - pi)

    def gen_curves(self):
        self.points.append(linspace(-pi, -pi/2, 300, endpoint=False))
        self.points.append(linspace(-pi/2, pi/2, 600, endpoint=False))
        self.points.append(linspace(pi/2, pi, 300))
        for set in self.points:
            for point in set:
                self.get_point(point)
        
    def plot(self):
        super().plot()
        plt.plot(self.points[0], self.graph[0], 'b')
        plt.plot([self.points[0][-1], self.points[1][0]], [self.graph[0][-1], self.graph[1][0]], 'b', linestyle='--')
        plt.plot(self.points[1], self.graph[1], 'b')
        plt.plot([self.points[1][-1], self.points[2][0]], [self.graph[1][-1], self.graph[2][0]], 'b', linestyle='--')
        plt.plot(self.points[2], self.graph[2], 'b')
        plt.savefig(self.name + '.png', format='png')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        print("./s_curve.py [BPSK|8PSK|16QAM]")
    else:
        if sys.argv[1] == 'BPSK':
            BPSK_data_aided().plot()
            BPSK_decision_directed().plot()
