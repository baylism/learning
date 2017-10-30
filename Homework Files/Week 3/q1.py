# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 17:09:36 2016

@author: Max
"""
M = 100
e = 2.71828
threshold = 0.05
N = 1500

probability_bound = 2 * M * e ** (-2 * threshold ** 2 * N)

print(probability_bound)
