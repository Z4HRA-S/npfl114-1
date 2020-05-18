#!/usr/bin/env python3

import numpy as np

if __name__ == "__main__":
    # Load data distribution, each line containing a datapoint -- a string.
    data_count = {}
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: Process the line, aggregating data with built-in Python
            # data structures (not NumPy, which is not suitable for incremental
            # addition and string mapping).
            data_count[line] = data_count[line] + 1 if line in data_count.keys() else 1

    # TODO: Create a NumPy array containing the data distribution. The
    # NumPy array should contain only data, not any mapping. If required,
    # the NumPy array might be created after loading the model distribution.
    total_count = np.sum(list(data_count.values()))
    data_dist = np.array([data_count[x] / total_count for x in data_count.keys()])

    # Load model distribution, each line `string \t probability`.
    model_dist = {}
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            # TODO: process the line, aggregating using Python data structures
            data_point, prob = line.split()
            model_dist[data_point] = float(prob)

    # TODO: Create a NumPy array containing the model distribution.
    for item in list(set(data_count.keys()) - set(model_dist.keys())):
        model_dist[item] = 0

    for item in list(set(model_dist.keys()) - set(data_count.keys())):
        data_count[item] = 0

    # TODO: Compute and print the entropy H(data distribution). You should not use
    # manual for/while cycles, but instead use the fact that most NumPy methods
    # operate on all elements (for example `*` is vector element-wise multiplication).
    entropy = -np.sum(data_dist * np.log(data_dist))
    print("{:.2f}".format(entropy))

    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
    cross_entropy = -np.sum([data_count[x] * np.log(model_dist[x]) for x in data_count.keys()])
    kl_divergence = cross_entropy - entropy
    print("{:.2f}".format(cross_entropy))
    print("{:.2f}".format(kl_divergence))
