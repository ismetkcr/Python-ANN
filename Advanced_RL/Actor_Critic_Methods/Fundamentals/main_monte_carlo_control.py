# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 16:21:56 2024

@author: ismt
"""

import gym
import matplotlib.pyplot as plt
from monte_carlo_control_agent import Agent

if __name__ == '__main__':
  env = gym.make('Blackjack-v1')
  agent = Agent(eps = 0.001)
  n_episodes = 100_000
  win_lose_draw = {-1:0, 0:0, 1:0}
  win_rates= []
  for i in range(n_episodes):
    if i>0 and i % 1000 == 0:
      pct = win_lose_draw[1] / i
      win_rates.append(pct)
    if i % 10_000 == 0:
      rates = win_rates[-1] if win_rates else 0.0
      print('starting episode', i, 'win_rates %.3f' % rates)
    observation = env.reset()[0]
    done = False
    while not done:
      action = agent.choose_action(observation)

      observation_, reward, done, trun, info = env.step(action)
      agent.memory.append((observation, action, reward))
      observation = observation_

    agent.update_Q()
    win_lose_draw[reward] += 1

  plt.plot(win_rates)
  plt.show()

