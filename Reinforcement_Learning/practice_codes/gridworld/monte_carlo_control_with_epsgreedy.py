# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 13:54:36 2024

@author: ismt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grid_world import standard_grid
from iterative_policy_evaluation_deterministic import print_values, print_policy

GAMMA = 0.9
ALL_POSIBLE_ACTIONS = ['U', 'D', 'L', 'R']

def epsilon_greedy(policy, s, eps=0.1):
  p = np.random.random()
  if p < (1-eps):
    return policy[s]
  else:
    return np.random.choice(ALL_POSIBLE_ACTIONS)

def play_game(grid, policy, max_steps = 20):
  #start state
  s = grid.reset()
  a = epsilon_greedy(policy, s)

  states = [s]
  actions = [a]
  rewards = [0]

  for _ in range(max_steps):
    r = grid.move(a)
    s = grid.current_state()

    rewards.append(r)
    states.append(s)

    if grid.game_over():
      break
    else:
      a = epsilon_greedy(policy, s)
      actions.append(a)

  return states, actions, rewards

def max_dict(d):
  max_val = max(d.values())
  max_keys = []
  for key, val in d.items():
    if val == max_val:
      max_keys.append(key)

  return np.random.choice(max_keys), max_val

if __name__ == '__main__':
  grid = standard_grid()

  print("rewards:")
  print_values(grid.rewards, grid)

  policy = {}
  for s in grid.actions.keys():
    policy[s] = np.random.choice(ALL_POSIBLE_ACTIONS)

  #init Q(s,a) and sample counts
  Q = {}
  sample_counts = {}
  state_sample_count = {}
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      Q[s] = {}
      sample_counts[s] = {}
      state_sample_count[s] = 0

      for a in ALL_POSIBLE_ACTIONS:
        Q[s][a] = 0
        sample_counts[s][a] = 0
    else:
      pass

  #repeat until convergence
  deltas = []
  for it in range(10_000):
    if it % 1000 == 0:
      print(it)

    biggest_change = 0
    states, actions, rewards = play_game(grid, policy)

    states_actions = list(zip(states, actions))

    T = len(states)
    G = 0
    for t in range(T-2, -1, -1):
      s = states[t]
      a = actions[t]

      #update G
      G = rewards[t+1] + GAMMA*G

      #first visit monte carlo
      if (s,a) not in states_actions[:t]:
        old_q = Q[s][a]
        sample_counts[s][a] += 1
        lr = 1 / sample_counts[s][a]
        Q[s][a] = old_q + lr * (G - old_q)

        #update polic
        policy[s] = max_dict(Q[s])[0]

        #update state_sample_count
        state_sample_count[s] += 1

        #bupdate delta
        biggest_change = max(biggest_change, np.abs(old_q - Q[s][a]))

    deltas.append(biggest_change)

  plt.plot(deltas)
  plt.show()

  print("final_policy:")
  print_policy(policy, grid)

  #find V
  V = {}
  for s, Qs in Q.items():
    V[s] = max_dict(Q[s])[1]

  print("final_values:")
  print_values(V, grid)

  state_sample_count_arr = np.zeros((grid.rows, grid.cols))
  for i in range(grid.rows):
    for j in range(grid.cols):
      if (i, j) in state_sample_count:
        state_sample_count_arr[i,j] = state_sample_count[(i, j)]


  df = pd.DataFrame(state_sample_count_arr)
  print(df)



