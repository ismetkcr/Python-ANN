import gym
import numpy as np
import pygame
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler

GAMMA = 0.9
ALPHA = 0.1
EPS = 0.1# Epsilon for epsilon-greedy
eps = 0.1

def gather_samples(env, n_episodes = 10000):
  samples = []
  for _ in range(n_episodes):
    s, info = env.reset()
    done = False
    truncated = False
    while not (done or truncated):
      a = env.action_space.sample()
      sa = np.concatenate((s, [a]))
      samples.append(sa)
      s, r, done, truncated, info = env.step(a)
  return samples


def getQs(model, s):
  #we need Q(s, a) to choose an action
  #i.e a = argmax[a] {Q(s,a)}
  Qs = {}
  for a in range(model.env.action_space.n):
        Qs[a] = model.predict(s, a)
  return Qs



def max_dict(d):
  max_key = None
  max_val = float('-inf')
  for key, val in d.items():
    if val > max_val:
      max_val = val
      max_key = key

  return max_key, max_val


def epsilon_greedy(model, a, eps):
  p = np.random.random()
  if p < (1-eps):
    return a
  else:
    return model.env.action_space.sample()

class Model:
  def __init__(self, env):
    self.env = env
    samples = gather_samples(env, n_episodes = 10_000)
    self.featurizer = RBFSampler()
    self.featurizer.fit(samples)
    dims = self.featurizer.n_components
    self.w = np.zeros(dims)

  def predict(self, s, a):
    sa = np.concatenate((s, [a]))
    x = self.featurizer.transform([sa])[0]
    return x @ self.w

  def grad(self, s, a):
    sa = np.concatenate((s, [a]))
    x = self.featurizer.transform([sa])[0]
    return x


def test_agent(model, env, n_episodes = 20):
  reward_per_episode = np.zeros(n_episodes)
  for it in range(n_episodes):
    done = False
    truncated = False
    episode_reward = 0
    s, info = env.reset()
    while not (done or truncated):
      Qs = getQs(model, s) # at state s we get all action values
      a = max_dict(Qs)[0] # we choose action regarding Qs
      s, r, done, truncated, info = env.step(a)
      episode_reward += r
    reward_per_episode[it] = episode_reward

  return np.mean(reward_per_episode)

def watch_agent(model, env, eps):
  done = False
  truncated = False
  episode_reward = 0
  s, info = env.reset()
  while not (done or truncated):
    Qs = getQs(model, s)
    a = max_dict(Qs)[0]
    a = epsilon_greedy(model, a, eps)
    s, r, done, truncated, info = env.step(a)
    episode_reward += r
    pygame.time.delay(50)
    if done or episode_reward >= 499:
      pygame.quit()

  print(f"Episode rewards -- > {episode_reward}")


if __name__ == '__main__':
  env = gym.make("CartPole-v1", render_mode = "rbg_array")
  model = Model(env)
  reward_per_episode = []

  #training Loop
  n_episodes = 1500
  for it in range(n_episodes):
    s, info = env.reset()
    episode_reward = 0
    done = False
    truncated = False

    #get qs for initial
    Qs = getQs(model, s)
    a = max_dict(Qs)[0]
    a = epsilon_greedy(model, a, eps)

    while not (done or truncated):
      s2, r, done, truncated, info = env.step(a)
      Qs2 = getQs(model, s2)
      a2 = max_dict(Qs2)[0]
      a2 = epsilon_greedy(model, a2, eps)

      #define target for sarsa
      if done:
        target = r
      else:
        #target = r + GAMMA * model.predict(s2, a2)
        target = r + GAMMA * Qs2[a2] #should be the same as up

      #update model
      g = model.grad(s, a)
      err = target - model.predict(s, a)
      model.w += ALPHA * err * g

      episode_reward += r
      s = s2
      a = a2

    if (it + 1) % 50 == 0:
      print(f"Episode: {it + 1}, Reward : {episode_reward}")

    reward_per_episode.append(episode_reward)

  #test trained agent
  test_reward = test_agent(model, env)
  print(f"Avarage test reward : {test_reward}")

  plt.plot(reward_per_episode)
  plt.title("rewards per eps")
  plt.show()

# Watch trained agent
  env = gym.make("CartPole-v1", render_mode="human")
  watch_agent(model, env, eps=0)






























