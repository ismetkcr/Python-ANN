# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:10:17 2024

@author: ismt
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def relu(x):
    return x * (x>0)

def sigmoid(A):
    return 1 / (1 + np.exp(-A))

def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()
    
def cost(T, Y):
  return -(T*np.log(Y)).sum()

def cost2(T, Y):
  #same as cost(), just uses the targets to index Y
  #instead of multiplyting by a large indicator matrix with mostly 0s
  # we use the actual values where as the targets are non zero actual values means  which are not one hot encoded
  #
  N = len(T)
  return -np.log(Y[np.arange(N), T]).sum()

def error_rate(targets, predictions):
  return np.mean(targets!=predictions)

def y2indicator(y):
  N = len(y)
  K = len(set(y))
  ind = np.zeros((N, K))
  for i in range(N):
    ind[i, y[i]] = 1
  return ind


def getData(balance_ones=True, Ntest=1000):
    # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    
    # shuffle and split
    X, Y = shuffle(X, Y)
    Xtrain, Ytrain = X[:-Ntest], Y[:-Ntest]
    Xvalid, Yvalid = X[-Ntest:], Y[-Ntest:]

    if balance_ones:
        # balance the 1 class
        X0, Y0 = Xtrain[Ytrain!=1, :], Ytrain[Ytrain!=1]
        X1 = Xtrain[Ytrain==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        Xtrain = np.vstack([X0, X1])
        Ytrain = np.concatenate((Y0, [1]*len(X1)))

    return Xtrain, Ytrain, Xvalid, Yvalid

def getData2(balance_ones=True, max_length=2304):
    # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            x_values = [int(p) for p in row[1].split()]
            # Pad or truncate the list to max_length
            x_values = x_values[:max_length] + [0] * (max_length - len(x_values))
            # Remove the last element
            x_values = x_values[:-1]
            X.append(x_values)

    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)


    return X / 255, Y

def getBinaryData():
    Y = []
    X = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)

class LogisticModelMulti(object):
  def __init__(self):
    pass

  def fit(self, X, Y, learning_rate = 10e-8, reg=10e-12, epochs=1000, show_fig=False):
    X, Y = shuffle(X, Y)
    Xvalid, Yvalid = X[-1000:], Y[-1000:]
    Tvalid = y2indicator(Yvalid)
    X, Y = X[:-1000], Y[:-1000]

    N, D = X.shape
    K = len(set(Y))
    T = y2indicator(Y)
    self.W = np.random.randn(D, K) / np.sqrt(D + K)
    self.b = np.zeros(K)

    costs = []
    best_validation_error = 1
    for i in range(epochs):
      pY = self.forward(X)

      #gradient descent
      self.W -= learning_rate*(X.T.dot(pY-T) + reg*self.W)
      self.b -= learning_rate*((pY-T).sum(axis=0) + reg*self.b)

      if i % 100 == 0:
        pYvalid = self.forward(Xvalid)
        c = cost(Tvalid, pYvalid)
        costs.append(c)
        e = error_rate(Yvalid, np.argmax(pYvalid, axis=1))
        print("i", i, "cost:", c, "error:", e)
        if e < best_validation_error:
          best_validation_error = e
    print("best_validation_error", best_validation_error)

    if show_fig:
      plt.plot(costs)
      plt.show()

  def forward(self, X):
    return softmax(X.dot(self.W) + self.b)

  def predict(self, X):
    pY = self.forward(X)
    return np.argmax(pY, axis=1)

  def score(self, X, Y):
    prediction = self.predict(X)
    return 1 - error_rate(Y, prediction)


X, Y = getData2()
#balance class 1
X0, Y0 = X[Y!=1, :], Y[Y!=1]
X1 = X[Y==1, :]
X1 =np.repeat(X1, 9, axis=0)
X = np.vstack([X0, X1])
Y = np.concatenate((Y0, [1]*len(X1)))

model = LogisticModelMulti()
model.fit(X, Y, show_fig=True)
model.score(X, Y)



