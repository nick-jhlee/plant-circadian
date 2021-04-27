"""
Created on 2021/04/25
@author: nicklee

(Description)
"""
import numpy as np
from pyswarms.single.global_best import GlobalBestPSO


def rosenbrock(x):
    a, b, c = 1, 100, 0
    # print((a - x[:,0])**2 + b*(x[:,1] - x[:,0])**2 + c)
    return (a - x[:, 0]) ** 2 + b * (x[:, 1] - x[:, 0]) ** 2 + c


options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options)

const, pos = optimizer.optimize(rosenbrock, 1000)
