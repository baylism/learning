# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 22:30:25 2016

@author: Max
"""
# TODO make create f depend on number of size of dataset
import numpy as np


def create_dataset(number_of_points):
    """Return dataset of random points in form x0=1, x1, x2"""
    ones = np.ones((number_of_points, 1))
    points = np.random.uniform(-1.0, 1.0, size=(number_of_points, 2))
    return np.concatenate((ones, points), axis=1)


def create_f(number_of_points):
    """Return coeficients of random straight line x0=1, m, c"""
    points = np.random.uniform(-1.0, 1.0, size=(number_of_points, 2))
    w0 = 1.0
    b = [-w0, -w0]
    w1, w2 = np.linalg.solve(points, b)
    return np.array([w0, w1, w2])


def evaluate_points(dataset, line):
    """Return list classifying points in dataset as above or below line"""

    return np.sign(dataset.dot(line))


def create_weights(dataset):
    """Return empty weight vector of appropriate size for dataset"""
    length = len(dataset[0])
    return np.zeros(length, int)


def error(point, weights, output):
    """Return gradient delta Ein for stochastic gradient descent"""
    return (-point * output) / (1 + np.e**(output * weights.dot(point)))


def epoch(dataset, output, weights, learning_rate):
    random_order = np.arange(100)
    np.random.shuffle(random_order)

    for point in random_order:
        point_error = error(dataset[point], weights, output[point])
        weights = weights - learning_rate * point_error

    return weights


def SGD(dataset, outputs, weights, learning_rate, stop):
    count = 0

    while True:
        old_weights = weights
        weights = epoch(dataset, outputs, weights, learning_rate)
        count += 1
        if np.linalg.norm(old_weights - weights) < stop:
            break

    return (count, weights)


def cross_entropy_error(point, weights, output):
    return np.log(1 + np.exp(-output * weights.dot(point)))


def out_of_sample_error(weights, target_function):
    dataset = create_dataset(1000)
    outputs = evaluate_points(dataset, target_function)
    errors = []
    for point in range(len(dataset)):
        error = cross_entropy_error(dataset[point], weights, outputs[point])
        errors.append(error)

    return errors


# create dataset of 100 points, create line, evaluate points
def run_SGD_experiment(runs):
    """Run SGD according to q8 spec"""

    iterations_needed = []
    out_of_sample_errors = []

    for run in range(runs):
        print("Run" + str(run))
        # Initialisations
        dataset = create_dataset(100)
        target_function = create_f(2)
        outputs = evaluate_points(dataset, target_function)
        weights = create_weights(dataset)

        # Run SGD
        result = SGD(dataset, outputs, weights, 0.01, 0.01)

        # Remember number of iterations require to complete SGD for this run
        iterations_needed.append(result[0])

        # Calculate and remember out of sample error for this run
        out_of_sample_errors.append(out_of_sample_error(result[1], target_function))

    return (sum(iterations_needed) / runs, np.mean(out_of_sample_errors))


print(run_SGD_experiment(10))
#
#
#dataset = create_dataset(1000)
#target_function = create_f(2)
#outputs = evaluate_points(dataset, target_function)
#weights = create_weights(dataset)
#
## Run SGD
#result = SGD(dataset, outputs, weights, 0.01, 0.01)
#
## Remember number of iterations require to complete SGD for this run
#iter = result[0]
#
## Calculate and remember out of sample error for this run
#err = out_of_sample_error(result[1], target_function)

