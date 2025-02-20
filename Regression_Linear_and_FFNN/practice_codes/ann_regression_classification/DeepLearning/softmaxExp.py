# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 15:48:08 2024

@author: ismt
"""
import numpy as np

a = np.random.randn(5)
expa = np.exp(a)

softmax_a = expa / expa.sum()

#for n sample
A = np.random.randn(20,5)
expA = np.exp(A)

nor_expA = expA / expA.sum(axis=1, keepdims=True)