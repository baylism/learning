# -*- coding: utf-8 -*-
"""
Created on Sun Oct  9 16:49:42 2016

@author: Max
"""

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
    result = np.zeros((coins, runs), dtype=np.int)

    for x in np.nditer(result, op_flags=['readwrite']):
        x[...] = np.random.randint(0, 2)

    return result


def v_1(results):
    """Return proportion of heads for first coin"""
    return np.mean(results[0])


def v_rand(results):
    """Return proportion of heads for first coin"""
    random_flip = np.random.randint(0, len(results))
    return np.mean(results[random_flip])


def v_min(results):
    """Return proportion of heads for coin with fewest heads"""
    head_frequencies = [np.mean(i) for i in results]
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

    v_1_average = np.mean(v_1_results)
    v_rand_average = np.mean(v_rand_results)
    v_min_average = np.mean(v_min_results)

    return v_1_average, v_rand_average, v_min_average

result = coin_experiment(100, 10, 100)
print(result)














