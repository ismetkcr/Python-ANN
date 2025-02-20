# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:59:47 2024

@author: ismt
"""
import numpy as np
# -- one hot encoding --

states = [(0, 0), (0, 1), (1, 0)]
ALL_POS_ACTS = ['U', 'D', 'L', 'R']
SA2IDX = {}
IDX = 0
#The loop that initializes SA2IDX will assign a unique index to each state-action pair.
for s in states:
  SA2IDX[s] = {}
  for a in ALL_POS_ACTS:
    SA2IDX[s][a] = IDX
    IDX += 1

"""
After this loop, SA2IDX will look like this:
  {
  (0, 0): {'U': 0, 'D': 1, 'L': 2, 'R': 3},
  (0, 1): {'U': 4, 'D': 5, 'L': 6, 'R': 7},
  (1, 0): {'U': 8, 'D': 9, 'L': 10, 'R': 11}
}
"""

def sa2x(s, a):
  # Create a zero vector of length IDX
  x = np.zeros(IDX)
  # Set the appropriate index to 1
  idx = SA2IDX[s][a]
  x[idx] = 1
  return x
## -- specific feature transform
# def sa2x(s, a):
#   return np.array([
#     s[0] - 1 if a == 'U' else 0,
#     s[1] - 1.5 if a == 'U' else 0,
#     (s[0] * s[1] - 3) / 3 if a == 'U' else 0,
#     (s[0] * s[0] - 2) / 2 if a == 'U' else 0,
#     (s[1] * s[1] - 4.5) / 4.5 if a == 'U' else 0,
#     1 if a == 'U' else 0,
#     s[0] - 1 if a == 'D' else 0,
#     s[1] - 1.5 if a == 'D' else 0,
#     (s[0] * s[1] - 3) / 3 if a == 'D' else 0,
#     (s[0] * s[0] - 2) / 2 if a == 'D' else 0,
#     (s[1] * s[1] - 4.5) / 4.5 if a == 'D' else 0,
#     1 if a == 'D' else 0,
#     s[0] - 1 if a == 'L' else 0,
#     s[1] - 1.5 if a == 'L' else 0,
#     (s[0] * s[1] - 3) / 3 if a == 'L' else 0,
#     (s[0] * s[0] - 2) / 2 if a == 'L' else 0,
#     (s[1] * s[1] - 4.5) / 4.5 if a == 'L' else 0,
#     1 if a == 'L' else 0,
#     s[0] - 1 if a == 'R' else 0,
#     s[1] - 1.5 if a == 'R' else 0,
#     (s[0] * s[1] - 3) / 3 if a == 'R' else 0,
#     (s[0] * s[0] - 2) / 2 if a == 'R' else 0,
#     (s[1] * s[1] - 4.5) / 4.5 if a == 'R' else 0,
#     1 if a == 'R' else 0,
#     1
# ])


#example function usage::
s = (0, 0)
a = 'D'
x = sa2x(s, a)