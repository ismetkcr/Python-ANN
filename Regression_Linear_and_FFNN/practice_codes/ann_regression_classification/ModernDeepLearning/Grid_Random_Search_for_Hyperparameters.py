# -*- coding: utf-8 -*-
"""
Created on Mon May 20 17:53:45 2024

@author: ismt
"""
import numpy as np

def accuracy(x,y,z):
  return x**2 + 5*y + (1 / z)

#grid search example
x_sizes = [0,10]
y_sizes = [1, 2, 3, 4]
z_sizes = [0.01, 0.1, 1]
print(z_sizes)

#loop through all possible parameters
best_accuracy = 0
best_x = None
best_y = None
best_z = None

for x in x_sizes:
  for y in y_sizes:
    for z in z_sizes:
      accuracy_rate = accuracy(x,y,z)
      print("accuracy: %.3f" % accuracy_rate)
      if accuracy_rate > best_accuracy:
        best_accuracy = accuracy_rate
        best_x = x
        best_y = y
        best_z = z
print("Best validation_accuracy:", best_accuracy)
print("Best parameters")
print("best x:", best_x)
print("best_y:", best_y)
print("best_z", best_z)

#random search-------------
#starting hyper parameters
x = 10
y = 1
z = 1
max_iters = 30
best_accuracy = 0
best_x = None
best_y = None
best_z = None

for _ in range(max_iters):
  accuracy_rate = accuracy(x,y,z)
  print("accuracy: %.3f" % accuracy_rate)
  if accuracy_rate > best_accuracy:
    best_accuracy = accuracy_rate
    best_x = x
    best_y = y
    best_z = z

    #select new hyper params
  x = best_x + np.random.randint(-10,20)
  y = best_y + np.random.randint(-10,20)
  z = best_z +np.random.randint(1,10)

print("Best validation_accuracy:", best_accuracy)
print("Best parameters")
print("best x:", best_x)
print("best_y:", best_y)
print("best_z", best_z)