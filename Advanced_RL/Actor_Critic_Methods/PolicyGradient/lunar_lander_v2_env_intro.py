# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:25:47 2024

@author: ismt
"""
import gym

if __name__ == '__main__':
  env = gym.make('LunarLander-v2', render_mode = 'human')
  n_games = 100

  for i in range(n_games):
    obs = env.reset()[0]
    score = 0
    done = False
    env.render()
    while not done:
      action = env.action_space.sample()
      obs_, reward, done, truncated,  info = env.step(action)
      score += reward
      env.render()
    print('epidode', i, 'score %.1f' % score)

