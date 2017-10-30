# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 20:09:01 2016

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


def create_weights(dataset):
    """Return empty weight vector of appropriate size for dataset"""
    length = len(dataset[0])
    return np.zeros(length, int)


def check_classifications(dataset, weights, y):
    """Return list of misclassified points in dataset"""
    misclassified_points = []

    for point_index in range(len(dataset)):
        if np.sign(dataset[point_index].dot(weights)) != y[point_index]:
            misclassified_points.append(point_index)

    return misclassified_points


def nudge(dataset, y, weights, misclassified_points):
    """Update weights using a random misclassified point"""

    point_index = np.random.choice(misclassified_points)

    weights = weights + y[point_index] * dataset[point_index]

    return weights


def linreg(dataset, y):
    """Return weights from linear regression"""
    pseudo_inverse = np.linalg.pinv(dataset)
    w = pseudo_inverse.dot(y)

    return w
