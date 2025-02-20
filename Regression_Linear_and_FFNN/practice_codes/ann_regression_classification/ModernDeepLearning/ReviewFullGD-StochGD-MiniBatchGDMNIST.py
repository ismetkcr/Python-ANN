# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:10:01 2024

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

Xtrain, Xtest, Ytrain, Ytest = get_normalized_data()
print("Performing logistic regression...")

N, D = Xtrain.shape
Ytrain_ind = y2indicator(Ytrain)
Ytest_ind = y2indicator(Ytest)

# 1. full Gradient Tüm datayı tekrar tekrar besliyoruz
W = np.random.randn(D, 10) / np.sqrt(D)
W0 = W.copy() # save for later
b = np.zeros(10)
test_losses_full = []
lr = 0.9
reg = 0.
t0 = datetime.now()
last_dt = 0
intervals = []

for i in range(50):
  p_y = forward(Xtrain, W, b)
  gW = gradW(Ytrain_ind, p_y, Xtrain) / N
  gb = gradb(Ytrain_ind, p_y) / N

  W += lr*(gW - reg*W)
  b += lr*(gb - reg*b)

  p_y_test = forward(Xtest, W, b)
  test_loss = cost(p_y_test, Ytest_ind)
  dt = (datetime.now() - t0).total_seconds()

  #save these
  dt2 = dt - last_dt
  last_dt = dt
  intervals.append(dt2)

  test_losses_full.append([dt, test_loss])
  if (i+1) % 10 == 0:
    print("Cost at iteration %d: %.6f" % (i+1, test_loss))

p_y = forward(Xtest, W, b)
print("Final error rate:", error_rate(p_y, Ytest))
print("Elapsed time for full GD:", datetime.now() - t0)

#save the max time so we dont surpass it in subsequent iterations
max_dt = dt
avg_interval_dt = np.mean(intervals)


#2. Stochastic GD dataları tek tek besliyoruz
W = W0.copy()
b = np.zeros(10)
test_losses_sgd = []
lr = 0.001
reg = 0.

t0 = datetime.now()
last_dt_calculated_loss = 0
done = False

for i in range(50):
  tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
  for n in range(N):
    x = tmpX[n,:].reshape(1,D)
    y = tmpY[n,:].reshape(1,10)
    p_y = forward(x, W, b)

    gW = gradW(y, p_y, x)
    gb = gradb(y, p_y)

    W += lr*(gW - reg*W)
    b += lr*(gb - reg*b)

    dt = (datetime.now() - t0).total_seconds()
    dt2 = dt - last_dt_calculated_loss

    if dt2 > avg_interval_dt:
      last_dt_calculated_loss = dt
      p_y_test = forward(Xtest, W, b)
      test_loss = cost(p_y_test, Ytest_ind)
      test_losses_sgd.append([dt, test_loss])

    #time to quit
    if dt > max_dt:
      done = True
      break
  if done:
    break


  if (i + 1) % 1 == 0:
    print("Cost at iteration %d: %.6f" % (i+1, test_loss))

p_y = forward(Xtest, W, b)
print("Final error rate:", error_rate(p_y, Ytest))
print("Elapsed time for SGD:", datetime.now() - t0)

#3. mini batch datayı batchler halinde besliyoruz

W = W0.copy()
b = np.zeros(10)
test_losses_batch = []
lr = 0.001
reg = 0.

t0 = datetime.now()
last_dt_calculated_loss = 0
done = False
batch_sz = 500
lr = 0.08
reg = 0.
n_batches = int(np.ceil(N / batch_sz))

t0 = datetime.now()
last_dt_calculated_loss = 0
done = False
for i in range(50):
  tmpX, tmpY = shuffle(Xtrain, Ytrain_ind)
  for j in range(n_batches):
    x = tmpX[j*batch_sz:(j + 1)*batch_sz, :]
    y = tmpY[j*batch_sz:(j + 1)*batch_sz, :]
    p_y = forward(x, W, b)

    current_batch_sz = len(x)
    gW = gradW(y, p_y, x) / current_batch_sz
    gb = gradb(y, p_y) / current_batch_sz

    W += lr*(gW - reg*W)
    b += lr*(gb - reg*b)

    dt = (datetime.now() - t0).total_seconds()
    dt2 = dt - last_dt_calculated_loss

    if dt2 > avg_interval_dt:
      last_dt_calculated_loss = dt
      p_y_test = forward(Xtest, W, b)
      test_loss = cost(p_y_test, Ytest_ind)
      test_losses_batch.append([dt, test_loss])

    #time to quit
    if dt > max_dt:
      done = True
      break
  if done:
    break

  if (i + 1) % 10 == 0:
    print("Cost at iteration %d: %.6f" % (i+1, test_loss))

p_y = forward(Xtest, W, b)
print("Final error rate:", error_rate(p_y, Ytest))
print("Elapsed time for mini-batch GD:", datetime.now() - t0)

#convert numpy arrays
test_losses_full = np.array(test_losses_full)
test_losses_sgd = np.array(test_losses_sgd)
test_losses_batch = np.array(test_losses_batch)

plt.plot(test_losses_full[:,0], test_losses_full[:,1], label="full")
plt.plot(test_losses_sgd[:,0], test_losses_sgd[:,1], label="sgd")
plt.plot(test_losses_batch[:,0], test_losses_batch[:,1], label="mini-batch")
plt.legend()
plt.show()









