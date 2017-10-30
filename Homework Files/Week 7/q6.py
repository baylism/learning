# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 11:10:40 2016

@author: Max
"""
import numpy as np


def validation_bias(runs):
    data = np.zeros((runs, 3))

    for row in data:
        row[0] = np.random.uniform(0.0, 1.0)
        row[1] = np.random.uniform(0.0, 1.0)
        row[2] = min(row[1], row[0])

    return data

runs = 100000
data = validation_bias(runs)

expected_e1 = np.mean(data[:, 0])
expected_e2 = np.mean(data[:, 1])
expected_min = np.mean(data[:, 2])

print("Runs: {0}".format(runs))
print("Expected e1 = {0}".format(expected_e1))
print("Expected e2 = {0}".format(expected_e2))
print("Expected min = {0}".format(expected_min))
