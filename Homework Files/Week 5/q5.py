# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 20:48:35 2016

@author: Max
"""

import numpy as np


def in_sample_error(x, y):
    """Return error from nonlinear error surface E(x, y)"""
    return ((x * (np.e ** y)) - (2 * y * (np.e ** -x))) ** 2


def partial_x(x, y):
    """Return partial derivative E(x, y) with respect to x"""
    return (2*np.e**y + 4.0*np.e**(-x)*y)*(np.e**y*x - 2*np.e**(-x)*y)


def partial_y(x, y):
    """Return partial derivative E(x, y) with respect to y"""
    return (np.e**y*x - 2*np.e**(-x)*y)*(2.0*np.e**y*x - 4*np.e**(-x))


def gradient_descent(x, y, learning_rate, target_error):
    """Run gradient descent according q5 spec"""
    count = 0
    error = in_sample_error(x, y)

    while error > target_error:
        x_temp = x
        x = x - learning_rate * partial_x(x, y)
        y = y - learning_rate * partial_y(x_temp, y)
        error = in_sample_error(x, y)
        count += 1

    return (count, x, y, error)

# Question 5
print(gradient_descent(1, 1, 0.1, 10 ** -14))


def coordinate_descent(x, y, learning_rate, iterations):
    """Run 'coordinate descent' according q6 spec"""

    error = in_sample_error(x, y)

    for x in range(iterations):
        x = x - learning_rate * partial_x(x, y)
        error = in_sample_error(x, y)
        y = y - learning_rate * partial_y(x, y)
        error = in_sample_error(x, y)

    return (x, y, error)

# Question 6
print(coordinate_descent(1, 1, 0.1, 15))

