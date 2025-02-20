import numpy as np
import matplotlib.pyplot as plt

lim = 2
x = np.linspace(-lim, lim, 100)
y = np.linspace(-lim, lim, 100)
xx, yy = np.meshgrid(x, y)
grid = np.vstack((xx.flatten(), yy.flatten())).T
xg = grid[:,0]
yg = grid[:,1]
f = xg ** 2 * yg ** 2 * np.exp(-xg**2 - yg**2)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(xg, yg, f, linewidth=0.2, antialiased=True)

def f(x_):
  x, y = x_
  return x ** 2 * y ** 2 * np.exp(-x**2 - y**2)

def grad(x_):
  x, y = x_
  g = 2 * x  * y  * np.exp(-x**2 - y**2) * np.array([y * (1 - x ** 2), x * (1 - y ** 2)])
  return g

# Gradient ascent
x_ = np.array([2,2])
n_iters = 300
vals = np.zeros(n_iters)
for i in range(n_iters):
  x_ = x_ + 0.1 * grad(x_)
  vals[i] = f(x_)

# Plot the values during gradient ascent
plt.figure()
plt.plot(vals)
plt.xlabel('Iterations')
plt.ylabel('Function Value')
plt.show()
