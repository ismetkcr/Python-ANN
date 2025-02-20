# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 14:23:08 2024

@author: ismt
"""

from utils2 import make_env

env = make_env('PongNoFrameskip-v4')
obs, info = env.reset()


for i in range(20):
    next_obs, reward, done, truncated, info = env.step(1)
    obs = next_obs



print('obs', obs.shape)
# Verify the shape
print('next_obs', next_obs.shape)
  
print('env',env.observation_space.low.shape)


