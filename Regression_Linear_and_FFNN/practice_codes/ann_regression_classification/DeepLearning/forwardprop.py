# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:10:30 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt

Nclass = 500 #samples
D = 2 #features
X1 = np.random.randn(Nclass, D) + np.array([0, -2]) #class1 x1, x2
X2 = np.random.randn(Nclass, D) + np.array([2, 2]) #class 2 x1, x2
X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
X = np.concatenate([X1, X2, X3], axis=0)
plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
plt.show();

M = 3 #hidden layer 
K = 3 # number of classes

W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def forward(X, W1, b1, W2, b2):
  Z = 1 / 1 + (np.exp(-X.dot(W1) - b1))
  A = Z.dot(W2) + b2
  expA = np.exp(A)
  Y = expA / expA.sum(axis=1, keepdims=True)
  return Y

def classification_rate(Y, P):
  n_correct = 0
  n_total = 0
  for i in range(len(Y)):
    n_total+=1
    if Y[i]==P[i]:
      n_correct += 1
  return float(n_correct/n_total)

P_Y_given_X = forward(X, W1, b1, W2, b2)
P = np.argmax(P_Y_given_X, axis=1)

print("Classification rate for randomly initialized weights: ", classification_rate(Y, P))
    

    



