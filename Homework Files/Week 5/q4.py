# -*- coding: utf-8 -*-
"""
Created on Sun Oct 30 18:52:58 2016

@author: Max
"""

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import mpmath as mp

u = sp.Symbol('u')
v = sp.Symbol('v')


print(sp.diff(((u*mp.e**v)-2*v*mp.e**-u)**2, u))
