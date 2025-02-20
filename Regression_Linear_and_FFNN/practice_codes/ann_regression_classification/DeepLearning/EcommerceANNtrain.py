# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 14:59:57 2024

@author: ismt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
#normalize first 2 columns
  for i in (1, 2):
    m = Xtrain[:, i].mean()
    s = Xtrain[:, i].std()
    Xtrain[:, i] = (Xtrain[:,i] - m) / s
    Xtest[:, i] = (Xtest[:,i] - m) / s

  return Xtrain, Ytrain, Xtest, Ytest

def y2indicator(y, K): #one hot encoded indicator matrix..
  N = len(y)
  ind = np.zeros((N, K))
  for i in range(N):
    ind[i, [y[i]]] = 1
  return ind


Xtrain, Ytrain, Xtest, Ytest = get_data()

D = Xtrain.shape[1]
K = len(set(Ytrain) | set(Ytest))

Ytrain_ind = y2indicator(Ytrain, K)
Ytest_ind = y2indicator(Ytest, K)

W = np.random.randn(D, K)
b = np.zeros(K)

def softmax(a):
  expA = np.exp(a)
  return expA / expA.sum(axis=1, keepdims=True)


def forward(X, W, b):
  return softmax(X.dot(W) + b)

def predict(P_Y_given_X):
  return np.argmax(P_Y_given_X, axis = 1)

def classification_rate(Y, P):
  return np.mean(Y==P)

def cross_entropy(Y, pY):
  return -np.mean(Y * np.log(pY)) #dön buraya

train_costs = []
test_costs = []
learning_rate = 0.001
for i in range(10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)
  
    ctrain = cross_entropy(Ytrain_ind, pYtrain)
    ctest = cross_entropy(Ytest_ind, pYtest)
  
    train_costs.append(ctrain)
    test_costs.append(ctest)
    
    #gradient..
    W -= learning_rate * Xtrain.T.dot(pYtrain-Ytrain_ind)
    b -= learning_rate * (pYtrain-Ytrain_ind).sum(axis=0)
    
    if i%1000==0:
        print(i, ctrain, ctest)


        


