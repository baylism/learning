# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 15:21:06 2016

@author: Max
"""

import numpy as np


def coin_flip(coins, runs):
    """Return list of random coin flips.

    coins -- number of coins to be flipped
    runs -- number of flips per coin
    """
    result = [[np.random.randint(0, 2) for run in range(runs)] for coin in range(coins)]
    return np.array(result)


def v_1(results):
    """Return proportion of heads for first coin"""
    return sum(results[0]) / len(results[0])


def v_rand(results):
    """Return proportion of heads for first coin"""
    random_flip = np.random.randint(0, len(results))
    return sum(results[random_flip]) / len(results[random_flip])


def v_min(results):
    """Return proportion of heads for coin with fewest heads"""
    head_frequencies = [sum(i) / len(i) for i in results]
    return min(head_frequencies)


def coin_experiment(coins, runs, repeats):
    v_1_results = []
    v_rand_results = []
    v_min_results = []

    for repeat in range(repeats):
        results = coin_flip(coins, runs)
        v_1_results.append(v_1(results))
        v_rand_results.append(v_rand(results))
        v_min_results.append(v_min(results))

    v_1_average = sum(v_1_results) / repeats
    v_rand_average = sum(v_rand_results) / repeats
    v_min_average = sum(v_min_results) / repeats

#    v_1_average = sum(v_1_results) / len(v_1_results)
#    v_rand_average = sum(v_rand_results) / len(v_rand_results)
#    v_min_average = sum(v_min_results) / len(v_min_results)

    return v_1_average, v_rand_average, v_min_average

result = coin_experiment(100, 10, 100)
print(result)















