# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:33:35 2024

@author: ismt
"""

import gym
from monte_carlo_predict_blackjack_agent import Agent




if __name__ == '__main__':
  env = gym.make('Blackjack-v1')
  agent = Agent()
  n_episodes = 250_000
  for i in range(n_episodes):
    if i % 5_000 == 0:
      print('starting episode', i)
    observation = env.reset()[0]
    done = False
    while not done:
      action = agent.policy(observation)
      prev_observation = observation

      observation, reward, done, truncated, info = env.step(action)
      agent.memory.append((prev_observation, reward))


    agent.update_V()

  print(agent.V[(21, 3, True)])
  print(agent.V[4, 1, False])


