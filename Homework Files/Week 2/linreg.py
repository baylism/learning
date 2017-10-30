# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 20:41:37 2016

@author: Max
"""

import numpy as np


def create_dataset(number_of_points):
    """Return dataset of random points in form x0=1, x1, x2"""
    ones = np.ones((number_of_points, 1))
    points = np.random.uniform(-1.0, 1.0, size=(number_of_points, 2))
    return np.concatenate((ones, points), axis=1)


def create_f(points):
    """Return coeficients of random straight line x0=1, m, c"""
    points = np.random.uniform(-1.0, 1.0, size=(points, 2))
    p0 = 1.0
    b = [-p0, -p0]
    w1, w2 = np.linalg.solve(points, b)
    return np.array([p0, w1, w2])


def evaluate_points(dataset, line):
    """Return list classifying points in dataset as above or below line"""

    return np.sign(dataset.dot(line))


def linreg(dataset, y):
    """Return weights from linear regression"""
    pseudo_inverse = np.linalg.pinv(dataset)
    w = pseudo_inverse.dot(y)

    return w


def calculate_error(dataset, weights, y):
    """Calculate error in weights"""
    output = evaluate_points(dataset, weights)
    comparison = np.equal(output, y)

    number_false = 0
    for c in comparison:
        if c == False:
            number_false += 1

    return number_false / len(y)
