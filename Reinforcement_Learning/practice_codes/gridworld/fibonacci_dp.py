# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 13:52:39 2024

@author: ismt
"""


#naive fibonacci
def fib(n):
  if n<= 2:
    result = 1
  else:
    result = fib(n-1) + fib(n-2)

  return result

#memoization fibonacci

memo = {}

def fib(n):
  if n in memo:
    return memo[n]

  if n <= 2:
    result = 1
  else:
    result = fib(n-1) + fib(n-2)

  memo[n] = result
  return result


#dynanic fib
def fib(n):
  memo = {}

  for i in range(1, n+1):
    if i<=2:
      result = 1
    else:
      result = memo[i-1] + memo[i-2]

    memo[i] = result

  return memo[n]



