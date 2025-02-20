# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 23:26:29 2024

@author: ismt
"""
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf



def manage_memory():
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    try:
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
      print(e)







def plot_learning_curve(x, scores, figure_file):
  running_avg = np.zeros(len(scores))
  for i in range(len(running_avg)):
    running_avg[i] = np.mean(scores[max(0, i-100), (i+1)])

  plt.plot(x, running_avg)
  plt.title('Running_averate of previos 100 scores')
  plt.savefig(figure_file)