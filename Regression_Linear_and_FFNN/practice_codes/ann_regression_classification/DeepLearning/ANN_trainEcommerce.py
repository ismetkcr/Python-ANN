# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 00:26:05 2024

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

  for i in (1, 2):
    m = Xtrain[:, i].mean()
    s = Xtrain[:, i].std()
    Xtrain[:, i] = (Xtrain[:,i] - m) / s
    Xtest[:, i] = (Xtest[:,i] - m) / s

  return Xtrain, Ytrain, Xtest, Ytest

def y2indicator(y, K):
  N = len(y)
  ind = np.zeros((N, K))
  for i in range(N):
    ind[i, y[i]] = 1
  return ind

Xtrain, Ytrain, Xtest, Ytest = get_data()

D = Xtrain.shape[1]
K = len(set(Ytrain) | set(Ytest)) #:) remember why
M = 5 # of neuron hidden layer

Ytrain_ind = y2indicator(Ytrain, K)
Ytest_ind = y2indicator(Ytest, K)

#randomly initialize weights
W1 = np.random.randn(D, M) # D dimensiyondan alıp M neouron içeren hidden layere
b1 = np.zeros(M)
W2 = np.random.randn(M, K) # M tane neouron içeren hidden layerden aldı K tane kategori içeren outputa
b2 = np.zeros(K)

def softmax(a):
  expA = np.exp(a)
  return expA / expA.sum(axis=1, keepdims=True)

def forward(X, W1, b1, W2, b2):
    Z = np.tanh(X.dot(W1)+ b1)
    return softmax(Z.dot(W2) + b2), Z

def predict(P_Y_given_X):
    return np.argmax(P_Y_given_X, axis=1)
    
def classification_rate(Y, P):
  return np.mean(Y==P)

def cross_entropy(Y, pY):
  return -np.mean(Y * np.log(pY))

train_costs = []
test_costs = []
train_accs = []
test_accs = []
learning_rate = 0.001
for i in range(10_000):
    pYtrain, Ztrain = forward(Xtrain, W1, b1, W2, b2)
    pYtest, Ztest = forward(Xtest, W1, b1, W2, b2)
    
    ctrain = cross_entropy(Ytrain_ind, pYtrain)
    ctest = cross_entropy(Ytest_ind, pYtest)
  
    train_costs.append(ctrain)
    test_costs.append(ctest)
  
    acc_train = classification_rate(Ytrain, predict(pYtrain))
    acc_test = classification_rate(Ytest, predict(pYtest))
    train_accs.append(acc_train)
    test_accs.append(acc_test)
    
    #gradient Descent
    gW2 = Ztrain.T.dot(pYtrain - Ytrain_ind)
    gb2 = (pYtrain - Ytrain_ind).sum(axis=0)
    dZ = (pYtrain - Ytrain_ind).dot(W2.T) * (1 - Ztrain * Ztrain)
    gW1 = Xtrain.T.dot(dZ)
    gb1 = dZ.sum(axis=0)
  
    W2 -= learning_rate * gW2
    b2 -= learning_rate * gb2
    W1 -= learning_rate * gW1
    b1 -= learning_rate * gb1
  
    if i % 1000 == 0:
      print(i, ctrain, ctest)
    
pYtrain, _ = forward(Xtrain, W1, b1, W2, b2)
pYtest, _ = forward(Xtest, W1, b1, W2, b2)
acc_train = classification_rate(Ytrain, predict(pYtrain))
acc_test = classification_rate(Ytest, predict(pYtest))
print("Final train classification rate:", acc_train)
print("Final test classification rate:", acc_test)

plt.plot(train_costs, label="train_cost")
plt.plot(test_costs, label="test_cost")
plt.legend()

























