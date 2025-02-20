# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 21:40:59 2024

@author: ismt
"""

def grid_paths(n, m):
  memo = {}
  for i in range(1, n+1):
    memo[(i, 1)] = 1
  for j in range(1, m+1):
    memo[1, j] = 1

  for i in range(2, n+1):
    for j in range(2, m+1):
      s = (i, j)
      memo[s] = memo[(i-1, j)] + memo[(i, j-1)]

  return memo[(n, m)]

print(grid_paths(3,5))