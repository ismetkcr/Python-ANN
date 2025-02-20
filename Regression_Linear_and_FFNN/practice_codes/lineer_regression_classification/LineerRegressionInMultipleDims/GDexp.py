# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:06:30 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3

X = np.zeros((N,D))
X[:,0] = 1 #bias..
X[:5,1] = 1 #first five element of second column
X[5:,2] = 1 #last five element of third column
Y = np.array([0]*5 + [1]*5)

print("X =  \n", X)
print("Y = \n", Y)

#GD.
costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001

for t in range(1000):
    Yhat = X.dot(w)
    delta = Yhat - Y
    w = w - learning_rate*X.T.dot(delta)
    mse = delta.dot(delta) / N
    costs.append(mse)
    
    
plt.plot(costs)
plt.show()