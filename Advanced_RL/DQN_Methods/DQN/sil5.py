# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 22:31:20 2024

@author: ismt
"""

import tensorflow as tf
import numpy as np



# Q-values for batch_size=2, n_actions=3
q_values = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
#actions taken [1,2] (in batch index 0, action 1; in batch idx 2, action 2)
actions = tf.constant([1,2])

#indices [batch_index, action] we want to extract
indices = tf.range(2)
print('indices', indices)
action_indices = tf.stack([indices, actions], axis=1)
print('action_indices', action_indices)

q_pred = tf.gather_nd(q_values, indices=action_indices)
print(q_pred.numpy)



val = np.array([[1.5, 2.3, 3.0, 0.5],
 [0.7, 1.1, 4.2, 2.8],
 [1.0, 2.5, 3.5, 4.0]])
