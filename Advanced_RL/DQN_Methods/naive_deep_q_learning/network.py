# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 15:44:30 2024

@author: ismt
"""

import tensorflow.keras as keras
import tensorflow as tf
import numpy as np




class LinearDeepQNetwork(keras.Model):
    def __init__(self, lr, n_actions):
        super(LinearDeepQNetwork, self).__init__()
        
        self.fc1 = keras.layers.Dense(128,activation='relu')
        self.fc2 = keras.layers.Dense(n_actions)
        
        self.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    
    def call(self, state):
        x = self.fc1(state)
        actions =self.fc2(x)
        return actions
    
class Agent:
    def __init__(self, n_actions, lr, gamma=0.99,
                 epsilon=1, eps_dec=1e-5, eps_min=0.01):
        self.lr = lr
        self.n_actions = n_actions
        self.gamma =gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        
        #init Q network
        self.Q = LinearDeepQNetwork(self.lr, n_actions)
    
    def choose_action(self, observation):
        if np.random.random()> self.epsilon:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            actions = self.Q(state)
            
            action = tf.argmax(actions, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
           
    def learn(self, state, action, reward, state_):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        action = tf.convert_to_tensor([action], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            q_pred = self.Q(state)[0]#predicted Q val
            
            q_pred = (q_pred[int(action)])
            q_next = tf.reduce_max(self.Q(next_state), axis=1)
            q_target = reward + self.gamma*q_next
            
            loss = tf.keras.losses.MSE(q_target, q_pred)
        params = self.Q.trainable_variables
        grads = tape.gradient(loss, params)
        self.Q.optimizer.apply_gradients(zip(grads, params))
        self.decrement_epsilon()