# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:24:19 2016

@author: Max
"""

import numpy as np

train = np.loadtxt("in.dta")
test = np.loadtxt("out.dta")


def nonlinear_transform(data):
    """Perform nonlinear transformation as per q2 spec"""
    result = []
    for row in data:
        x1 = row[0]
        x2 = row[1]

        result.append([1, x1, x2, np.multiply(x1, x1), np.multiply(x2, x2),
                       np.multiply(x1, x2), np.abs(x1 - x2), np.abs(x1 + x2)])

    return np.array(result)

def extract_labels(dataset):
    """Return correct classifications from dataset"""
    return dataset[:, 2]

def linreg(dataset, y):
    """Return weights from linear regression"""
    pseudo_inverse = np.linalg.pinv(dataset)
    w = pseudo_inverse.dot(y)

    return w

#def linreg_weight_decay(dataset, y, reg_factor):
#    """Return weights from linear regression with weight decay"""
##    n_col = dataset.shape[1]
##    return np.linalg.lstsq(dataset.T.dot(dataset) + reg_factor * np.identity(n_col), dataset.T.dot(y))
#
#    a = dataset.T.dot(dataset) + (np.identity(dataset.shape[1]) * reg_factor)
#    b = np.linalg.inv(a)
#    c = b.dot(dataset.T)
#    w = c.dot(y)
#
#    return w

def linreg_weight_decay(dataset, y, reg_factor):
    """Return weights from linear regression with weight decay"""
#    n_col = dataset.shape[1]
#    return np.linalg.lstsq(dataset.T.dot(dataset) + reg_factor * np.identity(n_col), dataset.T.dot(y))

    a = dataset.T.dot(dataset) + (np.identity(dataset.shape[1]) * reg_factor)
    b = np.linalg.inv(a)
    c = b.dot(dataset.T)
    w = c.dot(y)

    return w

def evaluate_points(dataset, line):
    """Return list classifying points in dataset as above or below line"""

    return np.sign(dataset.dot(line))

def calculate_error(dataset, weights, y):
    """Calculate error in weights"""
    output = evaluate_points(dataset, weights)
    comparison = np.equal(output, y)

    number_false = 0
    for c in comparison:
        if c == False:
            number_false += 1

    return number_false / len(y)

# Question 2
#data = nonlinear_transform(train)
#labels = extract_labels(train)
#lr = linreg(data, labels)
#er = calculate_error(data, lr, labels)
#
#out = nonlinear_transform(test)
#out_labels = extract_labels(test)
#out_er = calculate_error(out, lr, out_labels)

# Question 3
#data = nonlinear_transform(train)
#labels = extract_labels(train)
#lr = linreg_weight_decay(data, labels, np.power(10, -3))
#er = calculate_error(data, lr, labels)
#
#out = nonlinear_transform(test)
#out_labels = extract_labels(test)
#out_er = calculate_error(out, lr, out_labels)

# Question 4
data = nonlinear_transform(train)
labels = extract_labels(train)
lr = linreg_weight_decay(data, labels, np.power(10, 3))
er = calculate_error(data, lr, labels)

out = nonlinear_transform(test)
out_labels = extract_labels(test)
out_er = calculate_error(out, lr, out_labels)


# Question 5
for i in [2, 1, 0, -1, -2]:
    data = nonlinear_transform(train)
    labels = extract_labels(train)
    lr = linreg_weight_decay(data, labels, np.power(10, i))
    er = calculate_error(data, lr, labels)

    out = nonlinear_transform(test)
    out_labels = extract_labels(test)
    out_er = calculate_error(out, lr, out_labels)

    print(str(i) + ":  " + " er: " + str(er) + " out_er: " + str(out_er))

