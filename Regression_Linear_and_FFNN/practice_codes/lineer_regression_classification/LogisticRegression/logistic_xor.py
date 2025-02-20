# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 21:52:36 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt

N = 4
D = 2

#XOR
X = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]])

T = np.array([0, 1, 1, 0])
ones = np.ones((N,1))

#add column of x*y and 1 for bias
xy=(X[:,0]*X[:,1]).reshape(N,1)
Xb = np.column_stack((ones, xy, X))
#Xb2 = np.concatenate((ones, xy, X), axis=1)


w=np.random.randn(D+2)
z = Xb.dot(w)


def sigmoid(z):
    return 1/(1+np.exp(-z))


Y = sigmoid(z)

def cross_entropy(T,Y):
    E=0
    for i in range(len(T)):
        if T[i]==1:
            E -= np.log(Y[i])
        else:
            E -= np.log(1-Y[i])
    return E

learning_rate = 0.01
error = []
for i in range(10000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i%1000==0:
        print(e)
        
    w += learning_rate * (Xb.T.dot(T-Y) - 0.01*w)
    Y = sigmoid(Xb.dot(w))
    

plt.plot(error)
plt.title("Cross-entropy per iteration")
plt.show()

print("Final w:", w)
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(-1, 2, 0.1)
y = np.arange(-1, 2, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
zz = -1/w[1] * (w[2]*xx + w[3]*yy + w[0])
ax.plot_surface(xx, yy, zz, alpha=0.3)
ax.scatter(Xb[:,2], Xb[:,3], Xb[:,1], c=T)









