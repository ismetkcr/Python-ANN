# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 11:00:28 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt

N = 500
X = np.random.random((N, 2))*4 - 2
Y = X[:, 0] * X[:, 1]
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

#make networkkkk and train 
D = 2 #features..
M = 100 #number of hidden units..

#layer 1 
W1 = np.random.randn(D, M) / np.sqrt(D)
b1 = np.zeros(M)

#layer2
W2 = np.random.randn(M) / np.sqrt(M)
b2 = 0

def forward(X):
    Z = X.dot(W1) + b1
    Z = Z * (Z>0) # relu
    Yhat = Z.dot(W2) + b2
    return Z, Yhat

def derivative_W2(Z, Y, Yhat):
    return (Y - Yhat).dot(Z)

def derivative_W1(X, Z, Y, Yhat, W2):
    dZ = np.outer(Y - Yhat, W2) * (Z>0) #relu..
    return X.T.dot(dZ)

def derivative_b2(Y, Yhat):
    return (Y - Yhat).sum()

def derivative_b1(Z, Y, Yhat, W2):
    dZ = np.outer(Y - Yhat, W2) * (Z>0) #for relu
    return dZ.sum(axis=0)

def update(X, Z, Y, Yhat, W1, b1, W2, b2, learning_rate=1e-4):
  gW2 = derivative_W2(Z, Y, Yhat)
  gb2 = derivative_b2(Y, Yhat)
  gW1 = derivative_W1(X, Z, Y, Yhat, W2)
  gb1 = derivative_b1(Z, Y, Yhat, W2)

  W2 += learning_rate * gW2
  b2 += learning_rate * gb2
  W1 += learning_rate * gW1
  b1 += learning_rate * gb1

  return W1, b1, W2, b2

def get_cost(Y, Yhat):
  return ((Y - Yhat)**2).mean()

costs = []
for i in range(200):
    Z, Yhat = forward(X)
    W1, b1, W2, b2 = update(X, Z, Y, Yhat, W1, b1, W2, b2)
    cost = get_cost(Y, Yhat)
    costs.append(cost)
    if i%25==0:
      print(cost)

plt.plot(costs)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)

#surface plot
line = np.linspace(-2, 2, 20)
xx, yy = np.meshgrid(line, line)
Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
_, Yhat = forward(Xgrid)
ax.plot_trisurf(Xgrid[:, 0], Xgrid[:, 1], Yhat, linewidth=0.2, antialiased=True)
plt.show()













