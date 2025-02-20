# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 17:58:27 2024

@author: ismt
"""

import numpy as np
from grid_world import windy_grid, windy_grid_penalized, ACTION_SPACE
from iterative_policy_evaluation_deterministic import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9

def get_transition_probs_and_rewards(grid):
  transition_probs = {}
  rewards = {}

  for (s, a), v in grid.probs.items():
    for s2, p in v.items():
      transition_probs[(s, a, s2)] = p
      rewards[(s, a, s2)] = grid.rewards.get(s2, 0)

  return transition_probs, rewards

def evaluate_deterministic_policy(grid, policy, initV = None):
  #initialize V[s] = 0
  if initV is None:
    V = {}
    for s in grid.all_states():
      V[s] = 0
  else:
    V = initV

  #repeat until convergence
  it = 0
  while True:
    biggest_change = 0
    for s in grid.all_states():
      if not grid.is_terminal(s):
        old_v = V[s]
        new_v = 0
        for a in ACTION_SPACE:
          for s2 in grid.all_states():

            #action probability is deterministic
            action_prob = 1 if policy.get(s) == a else 0

            #rewards is a function of (s, a, s'), 0 if not specified
            r = rewards.get((s, a, s2), 0)
            new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + GAMMA*V[s2])

        #after done getting new value, update the value table
        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    it += 1

    if biggest_change < SMALL_ENOUGH:
      break
  return V

if __name__ == '__main__':

  #grid = windy_grid_penalized(-0.1)
  grid = windy_grid()
  transition_probs, rewards = get_transition_probs_and_rewards(grid)

  #print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  #we will randomly choose an action and update as we learn
  policy = {}
  for s in grid.actions.keys():
    policy[s] = np.random.choice(ACTION_SPACE)

  #initial policy
  print("initial policy:")
  print_policy(policy, grid)

  #repeat until convergence - break when policy does not change
  V = None
  while True:

    #policy evaluation step - we already now this
    V = evaluate_deterministic_policy(grid, policy, initV = V)

    #policy improvement step
    is_policy_converged = True
    for s in grid.actions.keys():
      old_a = policy[s]
      new_a = None
      best_value = float('-inf')

      #loop through all possible actions to find best current action
      for a in ACTION_SPACE:
        v = 0
        for s2 in grid.all_states():
          #reward is a function of (s, a, s'), 0 if not specified
          r = rewards.get((s, a, s2), 0)
          v += transition_probs.get((s, a, s2), 0) * (r + GAMMA*V[s2])

        if v > best_value:
          best_value = v
          new_a = a

      #new_a now represents the best action in this state
      policy[s] = new_a
      if new_a != old_a:
        is_policy_converged = False

    if is_policy_converged:
      break

  #once we are done, print final policy and values
  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)







