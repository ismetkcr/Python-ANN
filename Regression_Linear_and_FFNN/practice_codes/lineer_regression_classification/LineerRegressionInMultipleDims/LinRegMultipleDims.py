# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 22:14:53 2024

@author: ismt
"""
import numpy as np
import matplotlib.pyplot as plt

#load data
X = []
Y = []
for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([1, float(x1), float(x2)])
    Y.append(float(y))
    
X = np.array(X)
Y = np.array(Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0],X[:,1],Y)

w = np.linalg.solve(np.dot(X.T,X),np.dot(X.T,Y))
Yhat = np.dot(X,w)

#R-square
d1 = Y - Yhat
d2 = Y - Y.mean()
r2= 1 - d1.dot(d1) / d2.dot(d2)
print("r-squared", r2)