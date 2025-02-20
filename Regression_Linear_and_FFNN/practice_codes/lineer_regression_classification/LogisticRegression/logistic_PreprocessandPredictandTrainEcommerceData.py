# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 23:00:23 2024

@author: ismt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def get_data():
    df = pd.read_csv('ecommerce_data.csv')
    data = df.to_numpy()
    
    #datayı salla:D
    np.random.shuffle(data) #only rows..
    
    #split features and labels
    X = data[:, :-1]
    Y = data[:, -1]
    
    N, D = X.shape
    X2 = np.zeros((N, D+3)) #we have four categories.. and if we want to
    #one hot encode that we need four new columns..
    #but we can also replace the existing column, which is why we only need to add three More
    
    
    X2[:, :(D-1)] = X[:, :(D-1)] # non categorical columns..
    
    #üst satırda non categorical columns tanımlandı..
    #alt satırda categorical columns atandı...
    for n in range(N):
        t = int(X[n, D-1])
        X2[n, t+D-1] = 1
        
    X = X2
    
    Xtrain = X[:-100]
    Ytrain = Y[:-100]
    Xtest = X[-100:]
    Ytest = Y[-100:]
    
    #normalize column one and two..
    for i in (1,2):
        mtrain = Xtrain[:,i].mean()
        strain = Xtrain[:,i].std()
        Xtrain[:,i] = (Xtrain[:,i]-mtrain)/strain
        mtest = Xtest[:,i].mean()
        stest = Xtest[:,i].std()
        Xtest[:,i] = (Xtest[:, i] - mtest) / stest
        
    return Xtrain, Ytrain, Xtest, Ytest
    
# Xtrain, Ytrain, Xtest, Ytest = get_data()
# Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape

def get_binary_data():
    Xtrain, Ytrain, Xtest, Ytest = get_data()
    X2train = Xtrain[Ytrain<=1]
    Y2train = Ytrain[Ytrain<=1]
    X2test = Xtest[Ytest<=1]
    Y2test = Ytest[Ytest<=1]
    
    return X2train, Y2train, X2test, Y2test

Xtrain, Ytrain, Xtest, Ytest = get_binary_data()
#randomly initialize weights
D = Xtrain.shape[1]
W = np.random.randn(D)
b = 0
# X2train.shape, Y2train.shape, X2test.shape, Y2test.shape


def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def forward(X, W, b):
    return sigmoid(X.dot(W) + b)


def classification_rate(Y, P):
    return np.mean(Y == P)



## Train.. To achieve this, we need to define a function contains error, which called cost, lost, likelihood, ..

def cross_entropy(Y, pY):
    return -np.mean(Y*np.log(pY) + (1-Y)*np.log(1-pY))


train_cost = []
test_cost = []
learning_rate = 0.001

for i in range(10000):
    pYtrain = forward(Xtrain, W, b)
    pYtest = forward(Xtest, W, b)
    
    ctrain = cross_entropy(Ytrain, pYtrain)
    ctest = cross_entropy(Ytest, pYtest)
    train_cost.append(ctrain)
    test_cost.append(ctest)
    
    W -= learning_rate*Xtrain.T.dot(pYtrain-Ytrain)
    b -= learning_rate*(pYtrain - Ytrain).sum()
    if i % 1000 == 0:
        print(i, ctrain, ctest)
    
plt.plot(train_cost, label='train cost')
plt.plot(test_cost, label='test cost')
plt.legend()
plt.show()
    
