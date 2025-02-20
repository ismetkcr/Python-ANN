# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 22:55:49 2024

@author: ismt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation_deterministic import print_values, print_policy

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# Create a dictionary for mapping state-action pairs to indices
SA2IDX = {}
IDX = 0

def epsilon_greedy(model, s, eps=0.1):
  # we'll use epsilon-soft to ensure all states are visited
  # what happens if you don't do this? i.e. eps=0
  p = np.random.random()
  if p < (1 - eps):
    values = model.predict_all_actions(s)
    return ALL_POSSIBLE_ACTIONS[np.argmax(values)]
  else:
    return np.random.choice(ALL_POSSIBLE_ACTIONS)

def sa2x(s, a):
    # Create a zero vector of length IDX
    x = np.zeros(IDX)
    # Set the appropriate index to 1
    idx = SA2IDX[s][a]
    x[idx] = 1
    return x

# def sa2x(s, a):
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
#     # if we use SA2IDX, a one for encoding for every (s,a) pair
#     # in reality we wouldnot want to do this bc
#     # we have just as many params as before


#   ])


class Model:
    def __init__(self):
        # Initialize weights for all state-action pairs
        self.theta = np.zeros(IDX) #for one hot encoding
        #self.theta = np.zeros(25) / np.sqrt(25) #we define transform properties

    def predict(self, s, a):
        x = sa2x(s, a)
        return self.theta.dot(x)

    def predict_all_actions(self, s):
        return [self.predict(s, a) for a in ALL_POSSIBLE_ACTIONS]

    def grad(self, s, a):
        return sa2x(s, a)


if __name__ == '__main__':
    grid = negative_grid(step_cost=-0.1)

    # Create the state-action index mapping
    states = grid.all_states()
    for s in states:
        SA2IDX[s] = {}
        for a in ALL_POSSIBLE_ACTIONS:
            SA2IDX[s][a] = IDX
            IDX += 1

    model = Model()
    reward_per_episode = []
    state_visit_count = {}

    n_episodes = 20000
    for it in range(n_episodes):
        if (it + 1) % 100 == 0:
            print(it + 1)

        s = grid.reset()
        state_visit_count[s] = state_visit_count.get(s, 0) + 1
        episode_reward = 0
        while not grid.game_over():
            a = epsilon_greedy(model, s)
            r = grid.move(a)
            s2 = grid.current_state()
            state_visit_count[s2] = state_visit_count.get(s2, 0) + 1

            # Get the target
            if grid.game_over():
                target = r
            else:
                values = model.predict_all_actions(s2)
                target = r + GAMMA * np.max(values)

            # Update the model
            g = model.grad(s, a)
            err = target - model.predict(s, a)
            model.theta += ALPHA * err * g

            # Accumulate reward
            episode_reward += r

            # Update state
            s = s2

        reward_per_episode.append(episode_reward)

    plt.plot(reward_per_episode)
    plt.title("Reward per episode")
    plt.show()

    # Obtain V* and pi*
    V = {}
    greedy_policy = {}
    for s in states:
        if s in grid.actions:
            values = model.predict_all_actions(s)
            V[s] = np.max(values)
            greedy_policy[s] = ALL_POSSIBLE_ACTIONS[np.argmax(values)]
        else:
            V[s] = 0

    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(greedy_policy, grid)

    print("state_visit_count:")
    state_sample_count_arr = np.zeros((grid.rows, grid.cols))
    for i in range(grid.rows):
        for j in range(grid.cols):
            if (i, j) in state_visit_count:
                state_sample_count_arr[i, j] = state_visit_count[(i, j)]
    df = pd.DataFrame(state_sample_count_arr)
    print(df)
