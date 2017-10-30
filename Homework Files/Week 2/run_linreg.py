# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 20:43:31 2016

@author: Max
"""
from linreg import *

def linreg_test(N, repeats):
    results = []

    for repeat in range(repeats):
        dataset = create_dataset(N)
        target_function = create_f(2)
        y = evaluate_points(dataset, target_function)
        weights = linreg(dataset, y)
        error = calculate_error(dataset, weights, y)

        result = np.append(weights, error)

        results.append(result)

    return np.array(results)

def out_sample_error_2(weights, target_function, N):
    #make new points
    new_points = create_dataset(N)
    #classify new points
    target_function = create_f(2)
    y = evaluate_points(new_points, target_function)

    errors = []
    for row in weights:
        errors.append(calculate_error(new_points, row, y))

    return np.array(errors)

def out_sample_error(weights, target_function, N):
    """Return out of sample error for weights/target function, over N points"""

    new_points = create_dataset(N)
    y = evaluate_points(new_points, target_function)

    return calculate_error(new_points, weights, y)


def linreg_test_out(N, repeats):
    results = []

    for repeat in range(repeats):
        dataset = create_dataset(N)
        target_function = create_f(2)
        y = evaluate_points(dataset, target_function)

        weights = linreg(dataset, y)
        error = calculate_error(dataset, weights, y)
        out_error = out_sample_error(weights, target_function, 1000)

        result = []
        result.append(weights)
        result.append(error)
        result.append(out_error)

        results.append(result)




    return np.array(results)

w = linreg_test_out(100, 1000)

avg = np.mean(w, axis=0)

print(avg)

#
#lrweights = np.array(w[:,0:3])
#
#eout = out_sample_error(lrweights, 10000)
#print(np.mean(eout))
#
##print(lrweights)
#
##TESTING
#d = create_dataset(10)
#l = create_f(2)
#y = evaluate_points(d, l)
#w = linreg(d, y)
#er = in_sample_error(d, w, y)

