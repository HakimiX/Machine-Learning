import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook


def Example1():

    x, y = np.loadtxt('example.txt', delimiter=',', unpack=True)
    plt.plot(x,y, label='Loaded from file!')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()

