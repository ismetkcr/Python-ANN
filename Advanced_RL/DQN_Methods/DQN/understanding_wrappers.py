# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 21:43:27 2024

@author: ismt
"""
"""
#basic example
class myclass:
    def __init__(self,a):
        self.a = a
    def func(self):
        number = self.a
        print("hello")
        return number


class PrintDecorator:
    def __init__(self, obj, number_to_add):
        self.num = number_to_add
        self.obj = obj
    
    def func(self):
        original_number = self.obj.func()
        new_number = original_number + self.num
        print("modified works")
        return new_number
    
obj = myclass(2)
number = obj.func()
print(number) #2
modified_obj = PrintDecorator(obj, 5)
sum_number = modified_obj.func() #7
print(sum_number)
"""








#Base env
class SimpleEnv:
    def __init__(self):
        self.state = 0
    
    def reset(self):
        self.state = 0
        print("base environment reset")
        return self.state
    
    def step(self, action):
        self.state += action
        print(f"base environment step: new state = {self.state}")
        return self.state, self.state #obs, reward
#first wrapper, wrapped environment is base
class ActionDoubler:
    def __init__(self, env):
        self.env = env
    def reset(self):
        return self.env.reset() 
    def step(self, action):
        doubled_action = action *2
        print(f"ActionDoubler : Doubling action {action} --> {doubled_action} ")
        return self.env.step(doubled_action) # call the wrapped environment's step
#second wrapper, wrapped environment is actiondoubler+base
class RewardIncrementer:
    def __init__(self, env):
        self.env = env
    
    def reset(self):
        return self.env.reset() # pass reset to the wrapped environment
    
    def step(self, action):
        obs, reward = self.env.step(action) # get results from the wrapped environment
        reward += 1 #modify the reward
        print(f"rewardınc : Increasing reward to {reward}")
        return obs, reward


env = SimpleEnv()
env = ActionDoubler(env)
env = RewardIncrementer(env)

obs = env.reset() #this will call the base environments reset
print(f"Initial observation: {obs}\n")

obs, reward = env.step(3) #step with action=3
print(f"Observation: {obs}, Reward: {reward}\n")

        