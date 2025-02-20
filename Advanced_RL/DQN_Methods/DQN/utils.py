import collections
import cv2
import numpy as np
import matplotlib.pyplot as plt
import gym
import tensorflow as tf

def manage_memory():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def plot_learning_curve(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-20):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)

class ActionRepeater(gym.Wrapper):
    def __init__(self, env, repeat=4, clip_reward=False, no_ops=0, fire_first=False):
        super(ActionRepeater, self).__init__(env)
        self.repeat = repeat
        self.clip_reward = clip_reward
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        total_reward = 0
        done = False
        
        for _ in range(self.repeat):
            obs, reward, done, truncated, info = self.env.step(action)
            if self.clip_reward:
                reward = np.clip(np.array([reward]), -1, 1)[0]
            total_reward += reward
            if done:
                break
                
        return obs, total_reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        if self.no_ops > 0:
            no_ops = np.random.randint(self.no_ops) + 1
            for _ in range(no_ops):
                obs, _, done, _, _ = self.env.step(0)
                if done:
                    obs, info = self.env.reset(**kwargs)

        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            obs, _, _, _, _ = self.env.step(1)
            
        return obs, info

class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.repeat(repeat, axis=0),
            high=env.observation_space.high.repeat(repeat, axis=0),
            dtype=np.uint8)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self, **kwargs):
        self.stack.clear()
        observation, info = self.env.reset(**kwargs)
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        
        stacked = np.array(self.stack)  # Shape: (4, height, width, channels)
        return stacked, info

    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack)  # Shape: (4, height, width, channels)

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env=None):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[0], shape[1], 1)  # (84, 84, 1)

    def observation(self, obs):
        # obs shape: (4, height, width, channels)
        processed_frames = []
        for frame in obs:
            new_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            resized_screen = cv2.resize(new_frame, self.shape[:2],
                                      interpolation=cv2.INTER_AREA)
            processed_frames.append(resized_screen)
        
        processed_stack = np.array(processed_frames)  # Shape: (4, 84, 84)
        return processed_stack

class MaxLastTwoFrames(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(MaxLastTwoFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(84, 84, 1),  # Adjusted shape to (84, 84, 1)
            dtype=np.uint8)

    def observation(self, obs):
        # Take last two frames from the stack
        last_two_frames = obs[-2:]  # Shape: (2, 84, 84)
        # Return max of last two frames with shape (84, 84)
        max_frame = np.maximum(last_two_frames[0], last_two_frames[1])  # Shape: (84, 84)
        return np.expand_dims(max_frame, axis=-1)  # Add last dimension to get (84, 84, 1)


def make_env(env_name, shape=(84,84,1), repeat=4, clip_rewards=False,
             no_ops=0, fire_first=False):
    env = gym.make(env_name)
    
    # First: Add action repeater at the lowest level
    env = ActionRepeater(env, repeat, clip_rewards, no_ops, fire_first)
    
    # Second: Stack 4 frames
    env = StackFrames(env, repeat)
    
    # Third: Preprocess all 4 frames
    env = PreprocessFrame(shape, env)
    
    # Fourth: Take max of last 2 frames
    env = MaxLastTwoFrames(env)
    
    return env


