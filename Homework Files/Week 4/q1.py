# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 18:08:09 2016

@author: Max
"""
import matplotlib.pyplot as plt
import numpy as np

def vc0(N):
    threshold = 0.05
    dvc = 10

    return 4 * (2 * N ** (dvc)) * np.e ** (-(1/8) * threshold ** 2 * N)



def vc(N, delta, dvc):
    growth_function = 4 * (2 * N ** (dvc))

    return np.sqrt((8/N) * np.log(growth_function / delta))

for i in [400000, 420000, 440000, 460000, 480000]:

    print(str(i) + ":  " + str(abs(vc(i, 0.05, 10) - 0.05)))
