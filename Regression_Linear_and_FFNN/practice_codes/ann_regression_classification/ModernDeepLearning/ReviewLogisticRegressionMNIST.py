# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:51:31 2024

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

def forward(X, W, b):
    #softmax
    a = X.dot(W) + b
    expA = np.exp(a)
    y = expA  / expA.sum(axis=1, keepdims=True)
    
    return y

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

def gradW(t, y, X):
  return X.T.dot(t-y)


def gradb(t,y):
  return (t-y).sum(axis=0)


##----Linear BenchMark---#
Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
print("Performing Logistic Regression...")

#
N, D = Xtrain.shape
Ytrain_ind = y2indicator(Ytrain)
Ytest_ind= y2indicator(Ytest)
K = Ytrain_ind.shape[1]

W = np.random.randn(D, K) / np.sqrt(D)
b = np.zeros(K)
train_losses = []
test_losses = []
train_classification_rate = []
test_classification_rate = []
train_classification_errors = []
test_classification_errors = []

lr = 0.00003
reg = 0.0
n_iters = 100
for i in range(n_iters):
    p_y= forward(Xtrain, W, b)
    train_loss = cost(p_y, Ytrain_ind)
    train_losses.append(train_loss)

    train_err = error_rate(p_y, Ytrain)
    train_classification_errors.append(train_err)

    p_y_test = forward(Xtest, W, b)
    test_loss = cost(p_y_test, Ytest_ind)
    test_losses.append(test_loss)

    test_err = error_rate(p_y_test, Ytest)
    test_classification_errors.append(test_err)
    
    W += lr*(gradW(Ytrain_ind, p_y, Xtrain) - reg*W)
    b += lr*gradb(Ytrain_ind, p_y)
    
    if (i+1) % 10 == 0:
      print(f"Iter: {i+1}/{n_iters}, Train loss: {train_loss:.3f} ",
            f"Train error: {train_err:.3f}, Test loss: {test_loss:.3f}",
            f"Test err: {test_err:.3f}")

p_y = forward(Xtest, W, b)
print("Final error rate:", error_rate(p_y, Ytest))

plt.plot(train_losses, label="Train loss")
plt.plot(test_losses, label="Test loss")
plt.title("Loss per iteration")
plt.legend()
plt.show()

plt.plot(train_classification_errors, label='Train error')
plt.plot(test_classification_errors, label='Test error')
plt.title("Classification Error per Iteration")
plt.legend()
plt.show()    


#----------------------------------#



















   