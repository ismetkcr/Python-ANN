# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 13:14:55 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_policy, print_values
from monte_carlo_policy_evaluation import play_game

SMALL_ENOUGH = 1e-3
LEARNING_RATE = 0.001
GAMMA = 0.9

if __name__ == '__main__':
  grid = standard_grid()

  print("rewards")
  print_values(grid.rewards, grid)

  policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'U',
    (2, 1): 'L',
    (2, 2): 'U',
    (2, 3): 'L',
  }

  #initialize theta
  #our model is V_hat = theta.dot(x)
  #where x = [row, col, row*col, 1]
  theta = np.random.randn(4) / 2
  def s2x(s):
    return np.array([s[0] - 1, s[1] - 1.5, s[0]*s[1]-3, 1])

  #repeat until convergence
  deltas = []
  k = 1.0
  for it in range(20000):
    if it % 100 == 0:
      k += 0.01
    alpha = LEARNING_RATE / k
    #generate episode using pi
    biggest_change = 0
    states, rewards = play_game(grid, policy)
    states_and_returns = (list(zip(states, rewards)))
    seen_states = set()
    G = 0 # return
    T = len(states)
    for t in range(T-2, -1, -1):
      s = states[t]
      r = rewards[t+1]
      G = r + GAMMA * G # reward per episode
      if s not in states[:t]:
        old_theta = theta.copy()
        x = s2x(s)
        V_hat = theta.dot(x)
        theta += alpha*(G - V_hat)*x
        biggest_change = max(biggest_change, np.abs(old_theta - theta).sum())
        seen_states.add(s)

    deltas.append(biggest_change)


  plt.plot(deltas)
  plt.show()

  #obtain  predicted values
  V = {}
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      V[s] = theta.dot(s2x(s))
    else:
      V[s] = 0


  print("values")
  print_values(V, grid)
  print("policy")
  print_policy(policy, grid)






