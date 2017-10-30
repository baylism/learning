# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 20:13:20 2016

@author: Max
"""


from perceptron2 import *

def run_perceptron(number_of_points):
    """Return number of iterations to complete PLA for dataset"""
    dataset = create_dataset(number_of_points)

    line = create_f(2)

    y = evaluate_points(dataset, line)

    weights = create_weights(dataset)

    count = 0

    while True:
        misclassified_points = check_classifications(dataset, weights, y)

        if misclassified_points:

            weights = nudge(dataset, y, weights, misclassified_points)
            count += 1

        else: break
    print(weights)
    return count

def run_test(repeats, number_of_points):
    """Return mean number of iterations before PLA converges"""
    results = []

    for i in range(repeats):
        results.append(run_perceptron(number_of_points))

    return sum(results)/len(results)

print(run_test(10, 100))
