# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:34:44 2024

@author: ismt
"""

import tensorflow as tf
import tensorflow.keras as keras

class PolicyNetwork(keras.Model):
  def __init__(self, n_actions, fc1_dims = 256, fc2_dims = 256):
    super(PolicyNetwork, self).__init__()
    self.fc1_dims = fc1_dims
    self.fc2_dims = fc2_dims

    #build layers
    self.fc1 = keras.layers.Dense(self.fc1_dims, activation = 'relu')
    self.fc2 = keras.layers.Dense(self.fc2_dims, activation = 'relu')
    self.pi = keras.layers.Dense(n_actions, activation = 'softmax')


  def call(self, state):
    #forward pass
    x = self.fc1(state)
    x = self.fc2(x)
    pi = self.pi(x)


    return pi


