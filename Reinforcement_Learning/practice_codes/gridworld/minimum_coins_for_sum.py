# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 14:28:49 2024

@author: ismt
"""

def min_ignore_none(a,b):
  if a is None:
    return b
  if b is None:
    return a
  return min(a, b)

#naive approach

def minimum_coins(m, coins): # m is sum of coins
  if m == 0:
    answer = 0
  else:
    answer = None
    for coin in coins:
      subproblem = m - coin
      if subproblem < 0:
        #skip solutions where we try to reach [m] from a negative subproblem
        continue
      answer = min_ignore_none(
        answer,
        minimum_coins(subproblem, coins) + 1)
  return answer


#memoization

memo = {}

def minimum_coins_memo(m, coins):
  if m in memo:
    return memo[m]

  if m == 0:
    answer = 0
  else:
    answer = None
    for coin in coins:
      subproblem = m - coin
      if subproblem < 0:
        continue

      answer = min_ignore_none(answer, minimum_coins_memo(subproblem, coins) + 1)
  memo[m] = answer
  return answer

#bottom up app

def minimum_coins_bottom(m, coins):
  memo = {}
  memo[0] = 0
  for i in range(1, m+1):
    for coin in coins:
      subproblem = i - coin
      if subproblem < 0:
        continue
      memo[i] = min_ignore_none(memo.get(i), memo.get(subproblem) + 1)

  return memo[m]

#how many ways

from collections import defaultdict

def how_many_ways(m, coins):
  memo = defaultdict(lambda _: 0)
  memo[0] = 1
  for i in range(1, m+1):
    memo[i] = 0
    for coin in coins:
      subproblem = i - coin
      if subproblem < 0:
        continue

      memo[i] = memo[i] + memo[subproblem]

  return memo[m]



