# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 12:52:57 2024

@author: ismt
"""
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-2, 2, 100)
y = -0.5 * 9.8 * x ** 2 + 2 * x + 1

plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.plot(x,y)
plt.show()

#initial guess
x_old = -1
def func(x):
    return -0.5 * 9.8 * x ** 2 + 2 * x + 1

def derivative(x):
    return -0.5 * 9.8 * 2 * x + 2

#newton method algorithm
while True:
    x_new = x_old - func(x_old) / derivative(x_old)
    
    
    if np.abs(x_new - x_old) < 1e-10:
        break
    #assign updated value to old x
    x_old = x_new

def second_derivative(x):
    return -0.5 * 9.8 * 2

    
print("zero at:", x_new)
answer2 = x_new # save for later ..
 
#to find max of this func we can check for second derivative..
# the algorithm
while True:
  x_new = x_old - derivative(x_old) / second_derivative(x_old)
  

  # stop when the value of x hasn't changed too much
  if np.abs(x_new - x_old) < 1e-10:
    break

  # don't forget to reassign x_old for the next step!
  x_old = x_new

# print the final answer
print("max at:", x_new)


plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.grid()
plt.scatter([x_new], [func(x_new)], c='red')
#plt.scatter([answer1], [func(answer2)], c='blue')
#plt.scatter([answer2], [func(answer2)], c='blue')
plt.plot(x, y);




















