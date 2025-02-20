# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 13:49:15 2024

@author: ismt
"""
import numpy as np
import matplotlib.pyplot as plt

line = np.linspace(-5, 5, 100)
xx, yy = np.meshgrid(line, line)
grid = np.vstack((xx.flatten(), yy.flatten())).T
x_values = grid[:, 0]
y_values = grid[:, 1]

z_values = x_values ** 2 + y_values ** 2


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(
    x_values, y_values, z_values, linewidth=0.2, antialiased=True, alpha=0.2
)
ax.scatter([2],[2],[8], c='red')

# ∇f = (2x, 2y) yön belirtir, x ve y ye değerler vererek 1 birim ilerleyeceğiz..
# u* = ∇f(2,2) / |∇f(2,2)| = (1/√2, 1/√2) theta=0 durumu cos(theta)=1 max similarty
#2, 2 noktasından, (1/√2, 1/√2), birim vektor ilerleyeceğiz, gradyan yönünde

steepest_ascent_x = 2 + 1 / np.sqrt(2)
steepest_ascent_y = 2 + 1 / np.sqrt(2)

highest_f = steepest_ascent_x ** 2 + steepest_ascent_y ** 2
highest_f