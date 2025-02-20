# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 19:42:52 2024

@author: ismt
"""

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_values, print_policy
from monte_carlo_control_with_epsgreedy import max_dict

GAMMA = 0.9
ALPHA = 0.1
ALL_POS_ACTS = ['U', 'D', 'L', 'R']

SA2IDX = {}
IDX = 0

class Model:
  def __init__(self):
    self.theta = np.random.randn(IDX) / np.sqrt(IDX)
    #if we use SA2IDX, a one-hot encoding for every
    #s,a pair in reality we wouldnt want
    #self.theta = np.random.randn(IDX) / np.sqrt(IDX)


  def sa2x(self, s, a):
      # Create a zero vector of length IDX
      x = np.zeros(IDX)
      # Set the appropriate index to 1
      idx = SA2IDX[s][a]
      x[idx] = 1
      return x


  # def sa2x(self, s, a):
  #   return np.array([
  #     s[0]-1    if a == 'U' else 0,
  #     s[1]-1.5  if a == 'U' else 0,
  #     (s[0]*s[1]-3)/3    if a == 'U'else 0,
  #     (s[0]*s[0]-2)/2    if a == 'U' else 0,
  #     (s[1]*s[1]-4.5)/4.5  if a =='U' else 0,
  #     1          if a == 'U' else 0,
  #     s[0]-1    if a == 'D' else 0,
  #     s[1]-1.5  if a == 'D' else 0,
  #     (s[0]*s[1]-3)/3    if a == 'D'else 0,
  #     (s[0]*s[0]-2)/2    if a == 'D' else 0,
  #     (s[1]*s[1]-4.5)/4.5  if a =='D' else 0,
  #     1          if a == 'D' else 0,
  #     s[0]-1    if a == 'L' else 0,
  #     s[1]-1.5  if a == 'L' else 0,
  #     (s[0]*s[1]-3)/3    if a == 'L'else 0,
  #     (s[0]*s[0]-2)/2    if a == 'L' else 0,
  #     (s[1]*s[1]-4.5)/4.5  if a =='L' else 0,
  #     1          if a == 'L' else 0,
  #     s[0]-1    if a == 'R' else 0,
  #     s[1]-1.5  if a == 'R' else 0,
  #     (s[0]*s[1]-3)/3    if a == 'R'else 0,
  #     (s[0]*s[0]-2)/2    if a == 'R' else 0,
  #     (s[1]*s[1]-4.5)/4.5  if a =='R' else 0,
  #     1          if a == 'R' else 0,
  #     1
  #     #if we use SA2IDX, a one for encoding for every (s,a) pair
  #     #in reality we wouldnot want to do this bc
  #     #we have just as many params as before


  #   ])

  # x = np.zeros(len(self.theta))
  # idx = SA2IDX[s][a]
  # x[idx] = 1
  # return x

  def predict(self, s,a):
    x = self.sa2x(s, a)
    return self.theta.dot(x)

  def grad(self, s, a):
    return self.sa2x(s, a)

def getQs(model, s):
  #we need Q(s, a) to choose an action
  #i.e a = argmax[a] {Q(s,a)}
  Qs = {}
  for a in ALL_POS_ACTS:
    q_sa = model.predict(s, a)
    Qs[a] = q_sa
  return Qs

def random_action(a, eps):
  ALL_POS_ACTS = ['U', 'D', 'L', 'R']
  p = np.random.random()
  if p <= eps:
    return np.random.choice(ALL_POS_ACTS)
  else:
    return a


if __name__ == '__main__':
  grid = negative_grid(step_cost = -0.1)
  #no policy init, we will derive our policy from most recent Q

  states = grid.all_states()
  for s in states:
    SA2IDX[s] = {}
    for a in ALL_POS_ACTS:
      SA2IDX[s][a] = IDX
      IDX += 1

  model = Model()
  t = 1.0
  t2 = 1.0
  deltas = []
  for it in range(20_000):
    if it % 100 == 0:
      t += 10e-3
      t2 += 0.01
    if it % 1000 == 0:
      print("it", it)
    alpha = ALPHA / t2

    #instead of generating episode, we will play an episode withn this loop

    s = (2, 0)
    grid.set_state(s)

    #get Qs so we can choose first action
    Qs = getQs(model, s)
    #the first (s, r) tuples is the state we start in and 0
    a = max_dict(Qs)[0]
    a = random_action(a, 0.5 / t) #eps greedy
    biggest_change = 0
    while not grid.game_over():
      r = grid.move(a)
      s2 = grid.current_state()

      #we need next action as well since Q(s, a) depends
      #Q(s', a'), if s2 not in policy then ists a terminal state
      #all Q are 0
      old_theta = model.theta.copy()
      if grid.is_terminal(s2):
        model.theta += alpha*(r - model.predict(s, a))*model.grad(s, a)
      else:
        #not terminal
        Qs2 = getQs(model, s2)
        a2 = max_dict(Qs2)[0]
        a2 = random_action(a2, 0.5/t)
        #we will update Q(s, a) as we experince
        model.theta += alpha*(r + GAMMA*model.predict(s2, a2) - model.predict(s, a)) * model.grad(s, a)
        s = s2
        a = a2
        biggest_change = max(biggest_change, np.abs(old_theta - model.theta).sum())
      deltas.append(biggest_change)
  plt.plot(deltas)

# Extract and print the value function
  V = {}
  policy = {}
  states = grid.all_states()
  for s in states:
      if s in grid.actions:
          Qs = getQs(model, s)
          V[s] = max(Qs.values())
          policy[s] = max_dict(Qs)[0]
      else:
          V[s] = 0
          policy[s] = None

  print("Values:")
  print_values(V, grid)

  print("Policy:")
  print_policy(policy, grid)

















