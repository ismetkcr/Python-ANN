# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 21:55:44 2024

@author: ismt
"""

import numpy as np
import collections

class RepeatActionAndMaxFrameKnightOnline:
    def __init__(self, env, repeat=3, clip_reward=True, no_ops=0, fire_first=False):
        self.env = env
        self.repeat = repeat
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first
        
        # Get observation shape from the initial environment observation
        initial_obs = env.get_obs()
        self.shape = initial_obs.shape
        self.frame_buffer = np.zeros((2,) + self.shape, dtype=initial_obs.dtype)

    def step(self, action):
        total_reward = 0.0
        done = False

        for i in range(self.repeat):
            obs, reward, done, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(reward, -1, 1)
            total_reward += reward

            # Store the observation in the frame buffer
            idx = i % 2
            self.frame_buffer[idx] = obs

            if done:
                break

        # Take the pixel-wise maximum of the last two frames
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        # Reset the environment and initialize the frame buffer
        obs = self.env.reset(**kwargs)
        

        # Initialize both frames in the buffer with the initial observation
        self.frame_buffer[0] = obs
        self.frame_buffer[1] = obs

        return obs
    
    def __getattr__(self, name):
        # Forward attribute access to the underlying environment
        return getattr(self.env, name)


class StackFramesKnightOnline:
    def __init__(self, env, repeat):
        self.env = env
        self.repeat = repeat
        
        # Create a deque to hold the last `repeat` frames
        self.stack = collections.deque(maxlen=repeat)
        
        # Define the stacked observation shape based on the initial observation
        initial_obs = env.get_obs()
        
        # Remove the channel dimension (1) from the initial observation shape
        self.observation_shape = (initial_obs.shape[0], initial_obs.shape[1], repeat)

    def reset(self, **kwargs):
        self.stack.clear()
        observation = self.env.reset(**kwargs)
        # Reshape observation from (221, 221, 1) to (221, 221)
        #observation = observation.squeeze(-1)
        
        # Initialize the stack with the initial observation
        for _ in range(self.repeat):
            self.stack.append(observation)
        
        # Stack frames along the last axis to create (221, 221, repeat)
        stacked_frames = np.stack(self.stack, axis=0)
        return stacked_frames

    def step(self, action):
        # Take a step in the environment
        observation, reward, done, info = self.env.step(action)
        
        
        # Add the new observation to the stack
        self.stack.append(observation)
        
        # Stack frames along the last axis to create (221, 221, repeat)
        stacked_frames = np.stack(self.stack, axis=0)
        
        return stacked_frames, reward, done, info

    def __getattr__(self, name):
        # Forward attribute access to the underlying environment
        return getattr(self.env, name)