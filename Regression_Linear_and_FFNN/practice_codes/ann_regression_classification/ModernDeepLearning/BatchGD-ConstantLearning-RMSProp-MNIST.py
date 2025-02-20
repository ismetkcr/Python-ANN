# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:36:41 2024

@author: ismt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from datetime import datetime
import pandas as pd




def get_normalized_data():
    csv_file_path = r'C:\Users\ismt\Desktop\Python-ANN\ModernDeepLearning\train.csv'
    df = pd.read_csv(csv_file_path)
    data = df.to_numpy().astype(np.float32)
    X = data[:, 1:]
    Y = data[:, 0]
    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]
    
    mu = Xtrain.mean(axis=0)
    std = Xtrain.std(axis=0)
    
    np.place(std, std==0, 1) #anywhere where std is zero replace 1
    Xtrain = (Xtrain - mu) / std
    Xtest = (Xtest - mu) / std
  
    return Xtrain, Xtest, Ytrain, Ytest

def forward(X, W1, b1, W2, b2):
    #relu
    Z = X.dot(W1) + b1
    Z[Z<0]=0
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z
    
    
def derivative_W2(Z, T, Y):
    return Z.T.dot(Y-T)

def derivative_b2(T, Y):
    return (Y - T).sum(axis=0)

def derivative_W1(X, Z, T, Y, W2):
    return X.T.dot(((Y-T).dot(W2.T)*np.sign(Z)))

def derivative_b1(Z, T, Y, W2):
    return  ((Y-T).dot(W2.T)*np.sign(Z)).sum(axis=0)

def y2indicator(y):
    N = len(y)
    y = y.astype(np.int32)
    ind = np.zeros((N, 10))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

def cost(p_y, t):
    tot = t * np.log(p_y)
    return -tot.sum()

def predict(p_y):
    return np.argmax(p_y, axis=1)
  
def error_rate(p_y, t):
    prediction = predict(p_y)
    return np.mean(prediction != t)

#compare 2 scenarios:
#1. batch GD
#2. batch GD RMSPROP
#
max_iter = 20
print_period = 10

Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
lr = 0.00004
reg = 0.01

Ytrain_ind = y2indicator(Ytrain)
Ytest_ind = y2indicator(Ytest)

N, D = Xtrain.shape
batch_sz = 500
n_batches = N // batch_sz

M = 300
K = 10
W1 = np.random.randn(D, M) / 28
b1 = np.zeros(M)
W2 = np.random.randn(M, K) / np.sqrt(M)
b2 = np.zeros(K)

#save initial Weights
W1_0 = W1.copy()
b1_0 = b1.copy()
W2_0 = W2.copy()
b2_0 = b2.copy()

#Batch 1
losses_batch = []
errors_batch = []
for i in range(max_iter):
  for j in range(n_batches):
    Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
    Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
    pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

    #updates
    W2 -= lr*(derivative_W2(Z, Ybatch, pYbatch) + reg*W2)
    b2 -= lr*(derivative_b2(Ybatch, pYbatch) + reg*b2)
    W1 -= lr*(derivative_W1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1)
    b1 -= lr*(derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1)

    if j % print_period == 0:
      #calculate just for LL
      pY, _ = forward(Xtest, W1, b1, W2, b2)
      l = cost(pY, Ytest_ind)
      losses_batch.append(l)
      print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

      e = error_rate(pY, Ytest)
      errors_batch.append(e)
      print("Error rate:", e)

pY, _ = forward(Xtest, W1, b1, W2, b2)
print("Final error rate:", error_rate(pY, Ytest))

#RMS PROP
W1 = np.random.randn(D, M) / 28
b1 = np.zeros(M)
W2 = np.random.randn(M, K) / np.sqrt(M)
b2 = np.zeros(K)

losses_rms = []
errors_rms = []
lr0 = 0.001
cache_W2 = 0
cache_b2 = 0
cache_W1 = 0
cache_b1 = 0
decay_rate = 0.999
eps = 0.000001
for i in range(max_iter):
  for j in range(n_batches):
    Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
    Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]
    pYbatch, Z = forward(Xbatch, W1, b1, W2, b2)

    #updates

    gW2 = derivative_W2(Z, Ybatch, pYbatch) + reg*W2
    cache_W2 = decay_rate*cache_W2 + (1 - decay_rate)*gW2*gW2
    W2 -= lr0 * gW2 / (np.sqrt(cache_W2) + eps)

    gb2 = derivative_b2(Ybatch, pYbatch) + reg*b2
    cache_b2 = decay_rate*cache_b2 + (1 - decay_rate)*gb2*gb2
    b2 -= lr0*gb2 / (np.sqrt(cache_b2) + eps)

    gW1 = derivative_W1(Xbatch, Z, Ybatch, pYbatch, W2) + reg*W1
    cache_W1 = decay_rate*cache_W1 + (1 - decay_rate)*gW1*gW1
    W1 -= lr0 * gW1 / (np.sqrt(cache_W1) + eps)

    gb1 = derivative_b1(Z, Ybatch, pYbatch, W2) + reg*b1
    cache_b1 = decay_rate*cache_b1 + (1 - decay_rate)*gb1*gb1
    b1 -= lr0*gb1 / (np.sqrt(cache_b1) + eps)


    if j % print_period == 0:
      #calculate just for LL
      pY, _ = forward(Xtest, W1, b1, W2, b2)
      l = cost(pY, Ytest_ind)
      losses_rms.append(l)
      print("Cost at iteration i=%d, j=%d: %.6f" % (i, j, l))

      e = error_rate(pY, Ytest)
      errors_rms.append(e)
      print("Error rate:", e)

pY, _ = forward(Xtest, W1, b1, W2, b2)
print("Final error rate:", error_rate(pY, Ytest))

plt.plot(losses_rms, label='rms')
plt.plot(losses_batch, label='const')
plt.legend()
plt.show()