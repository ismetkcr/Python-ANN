# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:41:29 2024

@author: ismt
"""

import gym
import numpy as np
from agent_actor_critic import Agent



if __name__ == '__main__':
    
    # env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v1')
    agent = Agent(alpha=1e-4, n_actions=env.action_space.n)
    n_games = 1800
    record_video = False
    load_checkpoint = False

    # do a mkdir video if you want to record video of the agent playing
   

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation, _ = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode {} score {:.1f} avg score {:.1f}'.format(
              i, score, avg_score))
        
        
        
        
    
    
    