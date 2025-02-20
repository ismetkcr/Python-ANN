# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:35:24 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50
# uniformly distributed numbers between -5, +5
X = (np.random.random((N, D)) - 0.5)*10

true_w = np.array([1, 0.5, -0.5] + [0]*(D - 3))

Y = X.dot(true_w) + np.random.randn(N)*0.5

# perform gradient descent to find w
costs = [] # keep track of squared error cost
w = np.random.randn(D) / np.sqrt(D) # randomly initialize w
learning_rate = 0.001
l1 = 10.0 # Also try 5.0, 2.0, 1.0, 0.1 - what effect does it have on w?
for t in range(500):
  # update w
  Yhat = X.dot(w)
  delta = Yhat - Y
  w = w - learning_rate*(X.T.dot(delta) + l1*np.sign(w))

  # find and store the cost
  mse = delta.dot(delta) / N
  costs.append(mse)

# plot the costs
plt.plot(costs)
plt.show()

plt.plot(true_w, label='true w')
plt.plot(w, label='w_map')
plt.legend();
plt.show();