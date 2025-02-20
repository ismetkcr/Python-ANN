# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:38:17 2024

@author: ismt
"""
import numpy as np

#Derivation

#f(x) = x^2
#f'(x) = 2x
#evaluate at x = 5
h = 1e-10
x = 5
print('Estimate:', ((x+h)**2 - x**2) / h)
print('True', 2*x)



# f(x) = x^3
# f'(x) = 3x^2
# evaluate at x = 2
h = 1e-10
x = 2
print("Estimate:", ((x + h)**3 - x**3) / h)
print("True:", 3 * x**2)


#Integral
#f(x) = x^2
#F(x) = (1/3)*x^3
true_area = (1/3)*2**3-(1/3)*1**3
estimated_area = 0 #we will acumulate rectangles
x_values = np.linspace(1,2,1000)
for i in range(len(x_values)-1):
    x1 = x_values[i]
    x2 = x_values[i+1]
    width = x2-x1
    x_mid = (x2+x1) / 2
    height = x_mid ** 2 
    estimated_area += width*height
print("true area:", true_area)
print("estimated area:", estimated_area)
    
