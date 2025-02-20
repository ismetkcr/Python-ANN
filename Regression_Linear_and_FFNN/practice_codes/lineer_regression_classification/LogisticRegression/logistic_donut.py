# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 22:39:23 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt

N = 1000
D = 2

R_inner = 5
R_outer = 10

#distance from origin is radius + random normal
#angle theta is uniformly distributed between(0, 2pi)
R1 = np.random.randn(N//2) + R_inner
theta = 2*np.pi*np.random.random(N//2)
X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

R2 = np.random.randn(N//2) + R_outer
theta = 2*np.pi*np.random.random(N//2)
X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

X = np.concatenate([ X_inner, X_outer ])
T = np.array([0]*(N//2) + [1]*(N//2)) # labels: first 500 are 0, last 500 are 1

plt.scatter(X[:,0], X[:,1], c=T)
plt.show()


# add a column of ones
# ones = np.array([[1]*N]).T # old
ones = np.ones((N, 1))

# add a column of r = sqrt(x^2 + y^2)
#r = np.sqrt( (X * X).sum(axis=1) ).reshape(-1, 1)
r = np.zeros((N,1))
for i in range(N):
  r[i] = np.sqrt(X[i,:].dot(X[i,:]))
Xb = np.concatenate((ones, r, X), axis=1)
Xb.shape


# randomly initialize the weights
w = np.random.randn(D + 2)

# calculate the model output
z = Xb.dot(w)

def sigmoid(z):
    return 1/(1 + np.exp(-z))


Y = sigmoid(z)

# calculate the cross-entropy error
def cross_entropy(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()



# let's do gradient descent 100 times
learning_rate = 0.0001
error = []
for i in range(5000):
    e = cross_entropy(T, Y)
    error.append(e)
    if i % 500 == 0:
        print(e)

    # gradient descent weight udpate with L2 regularization
    w += learning_rate * ( Xb.T.dot(T - Y) - 0.1*w )

    # recalculate Y
    Y = sigmoid(Xb.dot(w))
    


plt.plot(error)
plt.title("Cross-entropy per iteration")
plt.show()


print("Final w:", w)
print("Final classification rate:", 1 - np.abs(T - np.round(Y)).sum() / N)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,1], X[:,0], r)
plt.show()


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.arange(-15, 15, 0.1)
y = np.arange(-15, 15, 0.1)
xx, yy = np.meshgrid(x, y, sparse=True)
z = w[0] + w[1]*np.sqrt(xx**2 + yy**2) + w[2]*xx + w[3]*yy
#z = w[0] + w[1]*(xx*yy) + w[2]*xx + w[3]*yy
zz = -1/w[1] * (w[2]*xx + w[3]*yy + w[0])
ax.scatter(X[:, 0], X[:, 1], r, c=T)
ax.plot_surface(xx, yy, zz)
