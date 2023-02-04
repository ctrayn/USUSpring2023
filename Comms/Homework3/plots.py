import matplotlib.pyplot as plt

def plot_and_show(y:list, x:list=None, title=None, xlabel=None, ylabel=None, grid=None):
    plt.figure()

    if not x is None:
        plt.plot(y)
    else:
        plt.plot(y,x)

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

    plt.show()