# -*- coding: utf-8 -*-
"""
Created on Sat Nov 19 15:31:31 2016

@author: Max
"""

import numpy as np
from sklearn import svm, metrics
from sklearn.model_selection import cross_val_score, KFold

# Load data from files
train_dataset = np.loadtxt("features.train")
test_dataset = np.loadtxt("features.test")

## Initialise normal training data
#train = train_dataset[:, 1:]
#train_target = train_dataset[:, 0]
#
## Initialise testing data
#test = test_dataset[:, 1:]
#test_target = test_dataset[:, 0]

def x_v(x, target):
    """
    Return classifier of type x-versus-all
    does not modify target
    """
    data = np.copy(target)
    for point in np.nditer(data, op_flags=['readwrite']):
        if point == x:
            point[...] = 1
        else:
            point[...] = -1
    return data


def one_v_one(x, y, dataset):
    """
    Return classifier of type x-versus-all
    does not modify target
    """
    result = []
    for row in dataset:
        if row[0] == x:
            result.append([1, row[1], row[2]])
        elif row[0] == y:
            result.append([-1, row[1], row[2]])
    return np.array(result)

def calculate_error(x, y):
    """Calculate error in weights"""
    comparison = np.equal(x, y)

    number_false = 0.0
    for c in comparison:
        if c == False:
            number_false += 1

    return number_false / len(y)

# Initialise 1v5 training data
train_dataset_tmp = one_v_one(1, 5, train_dataset)
train = train_dataset_tmp[:, 1:]
train_target = train_dataset_tmp[:, 0]

# Initialise testing data
test_dataset_tmp = one_v_one(1, 5, test_dataset)
test = test_dataset_tmp[:, 1:]
test_target = test_dataset_tmp[:, 0]

# Question 9
#in_sample_score = []
#out_sample_score = []
#
#for C in [0.01, 1, 100, 10**4, 10**6]:
#    classifier = svm.SVC(C=C, kernel='rbf', gamma=1.0)
#    classifier.fit(train, train_target)
#
#    in_sample_score.append(classifier.score(train, train_target))
#    out_sample_score.append(classifier.score(test, test_target))
#
#print('Training score: ', in_sample_score)
#print('Testing score: ', out_sample_score)


## Question 8
#results = []
#for i in range(10):
#    classifier = svm.SVC(C=0.001, kernel='poly', degree=2, gamma=1.0)
#    kf = KFold(n_splits=10)
#    results.append(np.mean(cross_val_score(classifier, train, train_target, cv=kf)))
#
#print(1- np.mean(results))


# Question 7
results = []
for i in range(10):
    scores = []
    for C in [0.0001, 0.001, 0.01, 0.1, 1]:
        classifier = svm.SVC(C=C, kernel='poly', degree=2, gamma=1.0)
        kf = KFold(n_splits=10)
        scores.append(np.mean(cross_val_score(classifier, train, train_target, cv=kf)))
    results.append(np.argmax(scores))

print(results)




## Question 6
#number_support_vectors_5 = []
#in_sample_score_5 = []
#out_sample_score_5 = []
#
#number_support_vectors_2 = []
#in_sample_score_2 = []
#out_sample_score_2 = []
#
#for C in [0.0001, 0.001, 0.01, 1]:
#    classifier = svm.SVC(C=C, kernel='poly', degree=5, gamma=1.0,
#                         decision_function_shape='ovo')
#    classifier.fit(train, train_target)
#
#    number_support_vectors_5.append((len(classifier.support_vectors_)))
#    in_sample_score_5.append(1- classifier.score(train, train_target))
#    out_sample_score_5.append(1- classifier.score(test, test_target))
#
#for C in [0.0001, 0.001, 0.01, 1]:
#    classifier = svm.SVC(C=C, kernel='poly', degree=2, gamma=1.0,
#                         decision_function_shape='ovo')
#    classifier.fit(train, train_target)
#
#    number_support_vectors_2.append((len(classifier.support_vectors_)))
#    in_sample_score_2.append(1 - classifier.score(train, train_target))
#    out_sample_score_2.append(1- classifier.score(test, test_target))
#
#
#print('SV 2: ', number_support_vectors_2)
#print('SV 5: ', number_support_vectors_5)
#
#print('Training score 2: ', in_sample_score_2)
#print('Training score 5: ', in_sample_score_5)
#
#print('Testing score 2: ', out_sample_score_2)
#print('Testing score 5: ', out_sample_score_5)

# Question 5
#number_support_vectors = []
#in_sample_score = []
#out_sample_score = []
#
#for C in [0.001, 0.01, 0.1, 1]:
#    classifier = svm.SVC(C=C, kernel='poly', degree=2, gamma=1.0)
#    classifier.fit(train, train_target)
#
#    number_support_vectors.append((len(classifier.support_vectors_)))
#    in_sample_score.append(classifier.score(train, train_target))
#    out_sample_score.append(classifier.score(test, test_target))
#
#print('SV: ', number_support_vectors)
#print('Training score: ', in_sample_score)
#print('Testing score: ', out_sample_score)

# Question 4
# Selected classifier for question 2 was '0 versus all'
# Selected classifier for question 2 was '1 versus all'
#classifier_1 = svm.SVC(C=0.01, kernel='poly', degree=2)
#target_tmp = x_v(0, train_target)
#classifier_1.fit(train, target_tmp)
#print('Number of support vectors for Q2: ', len(classifier_1.support_vectors_))
#
#classifier_2 = svm.SVC(C=0.01, kernel='poly', degree=2)
#target_tmp = x_v(1, train_target)
#classifier_2.fit(train, target_tmp)
#print('Number of support vectors for Q3: ', len(classifier_2.support_vectors_))




# Question 2
#classifier = svm.SVC(C=0.01, kernel='poly', degree=2)
#for x in [1, 3, 5, 7, 9]:
#    target_tmp = x_v(x, train_target)
#    classifier.fit(train, target_tmp)
#    print(x, classifier.score(train, target_tmp))

# Question 1
#classifier = svm.SVC(C=0.01, kernel='poly', degree=2)
#for x in [0, 2, 4, 6, 8]:
#    target_tmp = x_v(x, train_target)
#    classifier.fit(train, target_tmp)
#    print(x, classifier.score(train, target_tmp))

# Classifier types
# one-versus-one: one digit +1, another digit -1, rest disregarded
# one-versus-all: one digit +1, rest -1



# Testing
#classifier = svm.SVC(C=0.01, kernel='poly', degree=2)
#classifier.fit(train, train_target)
#
#predicted = classifier.predict(train)
#print(metrics.accuracy_score(train_target, predicted))