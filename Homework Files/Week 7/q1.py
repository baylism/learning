# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 17:15:27 2016

@author: Max
"""
import numpy as np

train = np.loadtxt("in.dta")
test = np.loadtxt("out.dta")

training = train[:25]
validation = train[25:]


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
    weights = pseudo_inverse.dot(y)

    return weights


def linreg_weight_decay(dataset, y, reg_factor):
    """Return weights from linear regression with weight decay"""
    a = dataset.T.dot(dataset) + (np.identity(dataset.shape[1]) * reg_factor)
    b = np.linalg.inv(a)
    c = b.dot(dataset.T)
    weights = c.dot(y)

    return weights


def evaluate_points(dataset, line):
    """Return list classifying points in dataset as above or below line"""

    return np.sign(dataset.dot(line))


def calculate_error(dataset, weights, y):
    """Calculate error in weights"""
    output = evaluate_points(dataset, weights)
    comparison = np.equal(output, y)

    number_false = 0.0
    for c in comparison:
        if c == False:
            number_false += 1

    return number_false / len(y)

# Question 5
#Eout for k=6 in Question 4 = 0.192
#Eout for k=6 in Question 2 = 0.084
# ==>> B

# Question 4
#training_labels = extract_labels(validation)
#training_transformed = nonlinear_transform(validation)
#
#validation_labels = extract_labels(training)
#validation_transformed = nonlinear_transform(training)
#
#test_labels = extract_labels(test)
#test_transformed = nonlinear_transform(test)
#
#validation_errors = {}
#out_sample_errors = {}
#
#for k in [3, 4, 5, 6, 7]:
#    training_data = training_transformed[:, :k+1]
#    validation_data = validation_transformed[:, :k+1]
#    test_data = test_transformed[:, :k+1]
#
#    regression_weights = linreg(training_data, training_labels)
#
#    validation_errors[k] = calculate_error(validation_data, regression_weights,
#                                           validation_labels)
#    out_sample_errors[k] = calculate_error(test_data, regression_weights,
#                                           test_labels)

# ==>> D

# Question 3
#training_labels = extract_labels(validation)
#training_transformed = nonlinear_transform(validation)
#
#validation_labels = extract_labels(training)
#validation_transformed = nonlinear_transform(training)
#
#validation_errors = {}
#
#for k in [3, 4, 5, 6, 7]:
#    training_data = training_transformed[:, :k+1]
#    validation_data = validation_transformed[:, :k+1]
#
#    regression_weights = linreg(training_data, training_labels)
#
#    validation_errors[k] = calculate_error(validation_data, regression_weights,
#                                           validation_labels)
#
#
# ==>> D

# Question 2
training_labels = extract_labels(training)
training_transformed = nonlinear_transform(training)

validation_labels = extract_labels(validation)
validation_transformed = nonlinear_transform(validation)

test_labels = extract_labels(test)
test_transformed = nonlinear_transform(test)

validation_errors = {}
out_sample_errors = {}

for k in [3, 4, 5, 6, 7]:
    training_data = training_transformed[:, :k+1]
    validation_data = validation_transformed[:, :k+1]
    test_data = test_transformed[:, :k+1]

    regression_weights = linreg(training_data, training_labels)

    validation_errors[k] = calculate_error(validation_data, regression_weights,
                                           validation_labels)
    out_sample_errors[k] = calculate_error(test_data, regression_weights,
                                           test_labels)

# ==>> E

# Question 1A
#training_labels = extract_labels(training)
#training_transformed = nonlinear_transform(training)
#validation_labels = extract_labels(validation)
#validation_transformed = nonlinear_transform(validation)
#
#
#validation_errors = {}
#for k in [3, 4, 5, 6, 7]:
#    training_data = training_transformed[:, :k+1]
#    validation_data = validation_transformed[:, :k+1]
#
#    regression_weights = linreg(training_data, training_labels)
#    validation_errors[k] = calculate_error(validation_data, regression_weights,
#                                           validation_labels)
#
# ==>> D

# Question 1
#validation_labels = extract_labels(validation)
#validation_transformed = nonlinear_transform(validation)
#
#
#validation_errors = {}
#for k in [3, 4, 5, 6, 7]:
#    data = validation_transformed[:, :k+1]
#    regression_weights = linreg(data, validation_labels)
#    validation_errors[k] = calculate_error(data, regression_weights,
#                                           validation_labels)
#


#er = calculate_error(data, lr, labels)
#
#out = nonlinear_transform(test)
#out_labels = extract_labels(test)
#out_er = calculate_error(out, lr, out_labels)

