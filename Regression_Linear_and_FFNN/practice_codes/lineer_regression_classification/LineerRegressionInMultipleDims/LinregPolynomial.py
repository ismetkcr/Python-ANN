# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:52:42 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

for line in open('data_poly.csv'):
    x, y = line.split(',')
    x=float(x)
    X.append([1, x, x**2])
    Y.append(float(y))
    
X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:,1], Y)
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Yhat = np.dot(X, w)

plt.scatter(X[:,1],Y)
plt.plot(sorted(X[:,1]), sorted(Yhat), linewidth=4)
