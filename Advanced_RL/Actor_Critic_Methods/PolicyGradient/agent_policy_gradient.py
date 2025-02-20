# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 13:51:46 2024

@author: ismt
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_probability as tfp
from network_policy_gradient import PolicyNetwork

class Agent:
  def __init__(self, alpha=3*1e-3, gamma = 0.99, n_actions = 4,
               fc1_dims=256, fc2_dims=256, chkpt_dir='C:/Users/ismt/Desktop/Python-ANN/AdvancedReinforcementLearning/Phill_udemy/PolicyGradient'):
    self.gamma = gamma
    self.lr = alpha
    self.reward_memory = []
    self.state_memory = []
    self.action_memory = []
    self.chkpt_dir = chkpt_dir
    self.policy_model = PolicyNetwork(n_actions=n_actions,
                                      fc1_dims=fc1_dims,
                                      fc2_dims=fc2_dims)
    self.policy_model.compile(optimizer=keras.optimizers.Adam(lr=self.lr))

  def save_models(self):
    self.policy_model.save(self.chkpt_dir + 'reinforce')
    print('......models saved succesfully......')

  def load_models(self):
    self.policy_model = keras.models.load_model(self.chkpt_dir + 'reinforce')

  def choose_action(self, observation):
    state = tf.convert_to_tensor([observation], dtype = tf.float32)
    probabilities = self.policy_model(state)

    action_probs = tfp.distributions.Categorical(probs=probabilities)
    action = action_probs.sample()

    return action.numpy()[0]

  def store_transition(self, observation, action, reward):
    self.state_memory.append(observation)
    self.action_memory.append(action)
    self.reward_memory.append(reward)

  def learn(self):
    actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
    rewards = np.array(self.reward_memory)
    G = np.zeros_like(self.reward_memory)
    for t in range(len(rewards)):
      G_t = 0
      discount = 1
      for k in range(t,len(rewards)):
        G_t += discount*rewards[k]
        discount *= self.gamma
      G[t] = G_t

    with tf.GradientTape() as tape:
      loss = 0
      for idx, (g, state) in enumerate(zip(G, self.state_memory)):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        probs = self.policy_model(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        log_prob = action_probs.log_prob(actions[idx])
        #print(log_prob)
        loss+= -g * tf.squeeze(log_prob)

      grads = tape.gradient(loss, self.policy_model.trainable_variables)
      self.policy_model.optimizer.apply_gradients(zip(grads, self.policy_model.trainable_variables))

      self.state_memory = []
      self.action_memory = []
      self.reward_memory = []













