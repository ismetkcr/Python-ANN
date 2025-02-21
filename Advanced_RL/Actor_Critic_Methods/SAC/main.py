
 
import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, manage_memory
from gym import wrappers

if __name__ == '__main__':
    manage_memory()
    env = gym.make('LunarLanderContinuous-v2')
    agent = Agent(input_dims=env.observation_space.shape, env=env,
                  n_actions=env.action_space.shape[0])
    n_games = 250
    render_video = False

    # do a mkdir video if you want to record video of the agent playing.
    if render_video:
        env = wrappers.Monitor(env, 'video',
                               video_callable=lambda episode_id: True,
                               force=True)
    filename = 'inverted_pendulum.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')

    for i in range(n_games):
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, truncated, info = env.step(action)
            score += reward
            agent.store_transition(observation, action, reward,
                                   observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode {} score {:.1f} avg_score {:.1f}'.
              format(i, score, avg_score))

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
