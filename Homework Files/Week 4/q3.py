# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 17:06:23 2016

@author: Max
"""

import matplotlib.pyplot as plt
import numpy as np

t1 = np.arange(-1.0, 1.0, 0.01)
plt.plot(t1, np.sin(np.pi * t1))

