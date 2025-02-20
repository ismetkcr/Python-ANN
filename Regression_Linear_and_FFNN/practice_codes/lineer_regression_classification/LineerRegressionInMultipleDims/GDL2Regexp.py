# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:36:58 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt


N = 50

# generate data
X = np.linspace(0,10,N)
Y = 0.5*X + np.random.randn(N)

#make outliers
Y[-1] += 30
Y[-2] +=30

#plot the data
plt.scatter(X, Y);

#add bias term
X = np.vstack([np.ones(N), X]).T
# plot the maximum likelihood solution
w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(w_ml)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml)
plt.show()

l2 = 1000.0
w_map = np.linalg.solve(l2*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Yhat_map = X.dot(w_map)
plt.scatter(X[:,1], Y)
plt.plot(X[:,1], Yhat_ml, label='maximum likelihood')
plt.plot(X[:,1], Yhat_map, label='map')
plt.legend()
plt.show()