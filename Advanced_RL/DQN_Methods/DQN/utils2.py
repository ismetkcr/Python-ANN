# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 22:32:28 2024

@author: ismt
"""

import collections
import cv2
import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

def manage_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

class RepeatActionAndMaxFrame(gym.Wrapper):
    def __init__(self, env=None, repeat=4, clip_reward=False, no_ops=0, fire_first=False):
        super(RepeatActionAndMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.shape
        self.frame_buffer = np.zeros((2,) + self.shape, dtype=np.uint8)  # Buffer to hold 2 frames
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        t_reward = 0.0
        done = False

        for i in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            t_reward += reward

            # Store the observation in the frame buffer (either at index 0 or 1)
            idx = i % 2
            self.frame_buffer[idx] = obs

            if done:
                break
        
        # Take the pixel-wise maximum of the last two observations
        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])

        return max_frame, t_reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        # Perform no-op actions if specified
        if self.no_ops > 0:
            no_ops = np.random.randint(self.no_ops) + 1
            for _ in range(no_ops):
                obs, _, done, _, _ = self.env.step(0)
                if done:
                    obs, info = self.env.reset(**kwargs)

        # Optionally, start with a "FIRE" action
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _, _ = self.env.step(1)

        # Initialize both frames in the buffer with the initial observation
        self.frame_buffer[0] = obs
        self.frame_buffer[1] = obs

        return obs, info



class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = shape[:2]  # Only take height and width
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=self.shape,  # Now just (84, 84) - no channels
            dtype=np.float32
        )

    def observation(self, obs):
        # Convert to grayscale and resize
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1] and return as 2D array
        new_obs = np.array(resized_screen, dtype=np.float32) / 255.0
        
        return new_obs  # Shape: (84, 84)


class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.repeat = repeat
        
        # Define the observation space for stacked frames
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(env.observation_space.shape[0], 
                  env.observation_space.shape[1], 
                  repeat),  # Shape: (84, 84, 4)
            dtype=np.float32
        )
        self.stack = collections.deque(maxlen=repeat)

    def reset(self, **kwargs):
        self.stack.clear()
        observation, info = self.env.reset(**kwargs)
        
        # Fill the stack with the initial observation
        for _ in range(self.repeat):
            self.stack.append(observation)
        
        # Stack frames along the last axis
        stacked_frames = np.stack(self.stack, axis=-1)
        return stacked_frames, info

    def observation(self, observation):
        self.stack.append(observation)
        # Stack frames along the last axis
        stacked_frames = np.stack(self.stack, axis=-1)
        return stacked_frames


def make_env(env_name, shape=(84, 84), repeat=4, clip_rewards=False, no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionAndMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)  # Returns (221, 221)
    env = StackFrames(env, repeat)     # Returns (84, 84, 4)
    return env
