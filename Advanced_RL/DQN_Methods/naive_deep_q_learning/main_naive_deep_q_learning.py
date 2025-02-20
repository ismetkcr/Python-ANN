# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 16:17:58 2024

@author: ismt
"""

from network import Agent
import gym
import numpy as np


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 10_000
    scores = []
    eps_history = []
    
    agent = Agent(n_actions = env.action_space.n,
                  lr = 0.0001)
    for i in range(n_games):
        score = 0
        done = False
        obs = env.reset()[0]
        while not done:
            action = agent.choose_action(obs)
            obs_,reward,done,truncated,info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, obs_)
            obs = obs_
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print('episode ', i, 'score %.1f avg score %.1f epsilon %.2f' %
                   (score, avg_score, agent.epsilon))
    
    
    
    