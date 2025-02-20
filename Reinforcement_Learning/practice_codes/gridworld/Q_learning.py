# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 23:03:31 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_values, print_policy
from monte_carlo_control import max_dict

gamma = 0.9
alpha = 0.1
all_pos_acts = ('U', 'D', 'L', 'R')

def epsilon_greedy(Q, s, eps = 0.1):
  p = np.random.random()
  if p < eps:
    return np.random.choice(all_pos_acts)
  else:
    a_opt = max_dict(Q[s])[0]
    return a_opt

if __name__ == '__main__':
  #grid = standard_grid()
  grid = negative_grid(step_cost=-0.2)

  print('rewards')
  print_values(grid.rewards, grid)

  #init Q(s a)
  Q = {}
  for s in grid.all_states():
    Q[s] = {}
    for a in all_pos_acts:
      Q[s][a] = 0
  #keep track of how many times Q[s] has been updated
  update_counts = {}

  reward_per_episode = []
  for it in range(10_000):
    if it % 2000 == 0:
      print("it:", it)

    #new eps start
    s = grid.reset()
    episode_reward = 0
    while not grid.game_over():
      #perfor act and get next state and reward
      a = epsilon_greedy(Q, s, eps = 0.1)
      r = grid.move(a)
      s2 = grid.current_state()

      episode_reward += r
      maxQ = max_dict(Q[s2])[1]
      Q[s][a] = Q[s][a] + alpha*(r + gamma*maxQ - Q[s][a])
      update_counts[s] = update_counts.get(s, 0) + 1
      s = s2
    reward_per_episode.append(episode_reward)
  plt.plot(reward_per_episode)
  plt.title("reward_per_episode")
  plt.show()

  # determine the policy from Q*
  # find V* from Q*
  policy = {}
  V = {}
  for s in grid.actions.keys():
    a, max_q = max_dict(Q[s])
    policy[s] = a
    V[s] = max_q

  # what's the proportion of time we spend updating each part of Q?
  print("update counts:")
  total = np.sum(list(update_counts.values()))
  for k, v in update_counts.items():
    update_counts[k] = float(v) / total
  print_values(update_counts, grid)

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)





