# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:49:54 2016

@author: Max
"""
import matplotlib.pyplot as plt
import numpy as np

def f(x):
    return x

t1 = np.arange(0.0, 500, 100)
t2 = np.arange(0.0, 5.0, 0.1)

plt.plot(t1, f(t1))