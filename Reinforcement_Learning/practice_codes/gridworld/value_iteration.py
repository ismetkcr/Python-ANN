# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 23:06:47 2024

@author: ismt
"""

import numpy as np
from iterative_policy_evaluation_deterministic import print_values, print_policy
from grid_world import windy_grid, ACTION_SPACE

SMALL_ENOUGH = 1e-3
GAMMA = 0.9

def get_transition_probs_and_rewards(grid):
  transition_probs = {}
  rewards = {}

  for (s,a), v in grid.probs.items():
    for s2, p in v.items():
      transition_probs[(s, a, s2)] = p
      rewards[(s, a, s2)] = grid.rewards.get(s2, 0)

  return transition_probs, rewards


if __name__ == '__main__':
  grid = windy_grid()
  transition_probs, rewards = get_transition_probs_and_rewards(grid)

  print("rewards:")
  print_values(grid.rewards, grid)

  #init V[s]
  V = {}
  states = grid.all_states()
  for s in states:
    V[s] = 0

  #repeat until convergence
  #V[s] = max[a]{sum[s', r]{p(s',r|s, a)[r + gamma*V[s]]}}

  it = 0
  while True:
    biggest_change = 0
    for s in grid.all_states():
      if not grid.is_terminal(s):
        old_v = V[s]
        new_v = float('-inf')

        for a in ACTION_SPACE:
          v = 0
          for s2 in grid.all_states():
            #rewards is a function of ((s, a, s2), 0)
            r = rewards.get((s, a, s2), 0)
            v += transition_probs.get((s, a, s2), 0) * (r + GAMMA*V[s2])
          #keep v if its better
          if v > new_v:
            new_v = v

        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))


    it += 1
    if biggest_change < SMALL_ENOUGH:
      break

  # find the policy that leads to optimal value function
  policy = {}
  for s in grid.actions.keys():
    best_a = None
    best_value = float('-inf')
    #loop through all possible actions to find the best current action
    for a in ACTION_SPACE:
      v = 0
      for s2 in grid.all_states():
        r = rewards.get((s, a, s2), 0)
        v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

      if v > best_value:
        best_value = v
        best_a = a

    policy[s] = best_a

  #our goal here is to verify that we get the same answer aas with polict iteration
  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)

