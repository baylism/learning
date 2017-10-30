# -*- coding: utf-8 -*-
"""
Created on Sun Nov 13 18:13:16 2016

@author: Max
"""
import numpy as np
import quadprog as qp


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


def compare_weights(weights_1, weights_2, runs):
    test_points = create_dataset(runs)
    labels_1 = evaluate_points(test_points, weights_1)
    labels_2 = evaluate_points(test_points, weights_2)
    print("l1: " + str(len(labels_1)))
    print("l2: " + str(len(labels_2)))

    differences = 0
    for point in range(runs):
        if labels_1[point] == labels_2[point]:
            differences += 1

    return differences / runs


def run_perceptron(number_of_points):
    """Return weights from PLA after all points classified correctly"""

    # Ensure all points not on same side of line

    while True:
        dataset = create_dataset(number_of_points)
        target_function = create_f(2)
        labels = evaluate_points(dataset, target_function)
        if not np.all(labels == labels[0]):
            break

    weights = create_weights(dataset)

    while True:
        misclassified_points = check_classifications(dataset, weights, labels)
        if misclassified_points:
            weights = nudge(dataset, labels, weights, misclassified_points)
        else:
            break

    return compare_weights(weights, target_function, 1000000)


def create_G(dataset, y):
    points = dataset[:, 1:]
    G = np.zeros((points.shape[0], points.shape[0]))
    for row in range(points.shape[0]):
        for col in range(points.shape[0]):
            val = (y[row] * y[col]) * points[row].dot(points[col])

            G[row][col] = val

    return G

def create_a(N):
    return np.full((N, 1), -1.)

def create_C(y, N):
    return np.hstack((-y.reshape((N, 1)), np.identity(N)))

def create_b(N):
    return np.full((N, 1), 0.)

def SVM(dataset, y):
    N = dataset.shape[0]
    G = create_G(dataset, y)
    a = create_a(N)
    C = create_C(y, N)
    b = create_b(N)

    return qp.solve_qp(G, a, C, b, meq=1)



dataset = create_dataset(10)
target_function = create_f(2)
labels = evaluate_points(dataset, target_function)

result = SVM(dataset, labels)


#x = run_perceptron(1000)
#print(x)