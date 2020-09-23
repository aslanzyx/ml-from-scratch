import numpy as np
from math import log


def entropy(x: np.ndarray):
    return sum([x_i * log(x_i) for x_i in x])

def entropy_raw(x: np.ndarray):
    return entropy(x/sum(x))


def entropy_bin(x: np.ndarray):
    bin_data = np.bincount(x)
    return entropy(bin_data)


def most(x: np.ndarray):
    rank = []
    rank.append([])
    highest_rank = 0
    for x_i in x:
        flag = False
        for i in range(highest_rank):
            if x_i in rank[i]:
                # remove from original rank
                rank[i].remove(x_i)
                flag = True
                # append to a new rank
                if i == highest_rank:
                    rank.append([x_i])
                    highest_rank += 1
                else:
                    rank[i+1].append(x_i)
        if not flag:
            rank[0].append(x_i)
    return rank[highest_rank][0]


def score(x: np.ndarray, label: int):
    retval = 0
    for x_i in x:
        if x_i == label:
            retval += 1
    return retval

