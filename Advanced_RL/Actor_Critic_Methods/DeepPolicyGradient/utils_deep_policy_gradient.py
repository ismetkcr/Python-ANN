# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 16:50:08 2024

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