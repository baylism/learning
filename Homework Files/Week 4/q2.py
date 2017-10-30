# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:37:35 2016

@author: Max
"""
import matplotlib.pyplot as plt
import numpy as np

def vc(N, delta, dvc):
    """Return original VC bound as a function of N"""
    growth_function = 4 * (2 * N ** (dvc))

    return np.sqrt((8/N) * np.log(growth_function / delta))


def rademacher_pentalty(N, delta, dvc):
    """Return Rademacher Penalty Bound as function of N"""
    a = np.sqrt((2 * np.log(2 * N * N ** dvc) / N ))
    b = np.sqrt((2 / N) * np.log(1 / delta))
    c = 1 / N

    return a + b + c

def parrondo(N):
    """Parrondo and Van den Broek solved for delta, as a function of N. dvc = 50"""
    return (- np.log(N ** 50) - 5.48064) / (2 - N)

t1 = np.arange(0.0, 10000, 50)

plt.plot(t1, vc(t1, 0.05, 50), 'r')
plt.plot(t1, rademacher_pentalty(t1, 0.05, 50), 'g')
plt.plot(t1, parrondo(t1), 'b')
#plt.ylim(-10, 50)
