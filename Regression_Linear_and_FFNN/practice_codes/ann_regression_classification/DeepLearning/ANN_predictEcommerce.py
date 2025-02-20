# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:15:50 2024

@author: ismt
"""

import numpy as np
import pandas as pd

def get_data():
  df = pd.read_csv('ecommerce_data.csv')
  data = df.to_numpy()
  np.random.shuffle(data)

  X = data[:, :-1]
  Y = data[:, -1].astype(np.int32)

  N, D = X.shape
  X2 = np.zeros((N, D+3))
  X2[:, :(D-1)] = X[:, :(D-1)]

  for n in range(N):
    t = int(X[n, D-1])
    X2[n, t+D-1] = 1

  #more efficient to compose categorical input matrix
  #Z = np.zeros((N,4))
  #Z[np.arange(N), X[:, D-1].astype(np.int32)] = 1
  #Z[(r1, r2, r3, ...), (c1, c2, c3, ...)] = value
  #X2[:, -4] = Z

  X = X2
  Xtrain = X[:-100]
  Ytrain = Y[:-100]

  Xtest = X[-100:]
  Ytest = Y[-100:]

  for i in (1, 2):
    m = Xtrain[:, i].mean()
    s = Xtrain[:, i].std()
    Xtrain[:, i] = (Xtrain[:,i] - m) / s
    Xtest[:, i] = (Xtest[:,i] - m) / s

  return Xtrain, Ytrain, Xtest, Ytest


X, Y, _, _ = get_data()

#randomly initialize weights
M = 5
D = X.shape[1]
K = len(set(Y)) # categorical input

W1 = np.random.randn(D, M)
b1 = np.zeros(M)
W2 = np.random.randn(M, K)
b2 = np.zeros(K)

def softmax(a):
    expA = np.exp(a)
    return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1) + b1)
    return softmax(Z.dot(W2) + b2)

P_Y_given_X = forward(X, W1, b1, W2, b2)
P_Y_given_X.shape #probabilities

predictions = np.argmax(P_Y_given_X, axis=1)


def classification_rate(Y, P):
  return np.mean(Y==P)

print("classification rate is:", classification_rate(Y, predictions))


