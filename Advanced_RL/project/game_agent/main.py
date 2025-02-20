import numpy as np
from agent import Agent
from KnightOnlineEnv_class import KnightOnline
from env_wrappers import RepeatActionAndMaxFrameKnightOnline, StackFramesKnightOnline
import cv2
import keyboard

def toggle_pause():
    global pause
    pause = not pause
    if pause:
        print("Program paused. Press 'p' to continue.")
    else:
        print("Program continuing...")

keyboard.add_hotkey('p', toggle_pause)

if __name__ == '__main__':
    env = KnightOnline()
    env = RepeatActionAndMaxFrameKnightOnline(env)
    env = StackFramesKnightOnline(env, 3)  # repeat is 3
    best_score = -np.inf
    load_checkpoint = False
    record_agent = False
    pause = False

    n_games = 150
    agent = Agent(gamma=0.99, epsilon=0.5, lr=0.0001,
                  input_dims=(221, 221, 3),  # Changed to channels_last format
                  n_actions=env.actions, mem_size=2000, eps_min=0.01,
                  batch_size=32, replace=500, eps_dec=5e-4,
                  chkpt_dir='models/', algo='DQNAgent',
                  env_name='KnightOnline_v1')

    if load_checkpoint:
        agent.load_models()
        agent.epsilon = agent.eps_min

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    # Create window once
    cv2.namedWindow('Computer Vision', cv2.WINDOW_NORMAL)

    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    try:
        for i in range(n_games):
            done = False
            observation = env.reset()
            score = 0

            while not done:
                while pause:
                    # Keep updating the window even when paused
                    cv2.imshow('Computer Vision', observation[:, :, :3])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        raise KeyboardInterrupt
                    pass
           

                action = agent.choose_action(observation)
                print(f"chosen action_idx: {action}")
                observation_, reward, done, info = env.step(action)

                # # Show the current frame
                # cv2.imshow('Computer Vision', observation_[:, :, :3])
                # # Add small wait and check for 'q' key
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     raise KeyboardInterrupt

                score += reward
                print(f"chosen action_idx: {action} | current score in episode: {env.total_score} | current epsilon: {agent.epsilon:.4f} | current step: {n_steps}")
                print(f"current score in episode is clipped {score} | current episode step counter : {env.episode_step_counter}")
                

                if not load_checkpoint:
                    agent.store_transition(observation, action, reward, observation_, done)
                    agent.learn()

                observation = observation_
                n_steps += 1

            scores.append(score)
            steps_array.append(n_steps)
            avg_score = np.mean(scores[-10:])
            print(f'episode {i} score {score:.1f} avg score {avg_score:.1f} best score {best_score:.1f} epsilon {agent.epsilon:.2f} steps {n_steps}')

            if score > best_score:
                if not load_checkpoint:
                    agent.save_models()
                best_score = score

            eps_history.append(agent.epsilon)

    except KeyboardInterrupt:
        print("\nExiting program...")
    finally:
        # Clean up
        cv2.destroyAllWindows()
        keyboard.unhook_all()
