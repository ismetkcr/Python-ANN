import numpy as np
from agent import Agent
from KnightOnlineEnv_class import KnightOnline
from env_wrappers import RepeatActionAndMaxFrameKnightOnline, StackFramesKnightOnline
import cv2
import keyboard
import time  # Import time module to manage sleep intervals

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
    env = StackFramesKnightOnline(env, 3)
    best_score = 20
    load_checkpoint = True
    resume_training = True
    pause = False

    n_games = 750
    agent = Agent(gamma=0.99, epsilon=0.1, lr=0.0002,
                  input_dims=(3, 128, 128, 3),
                  n_actions=env.actions, mem_size=5000, eps_min=0.01,
                  batch_size=32, replace=500, eps_dec=5e-4,
                  chkpt_dir='models/', algo='DQNAgent',
                  env_name='KnightOnline_v1')

    if load_checkpoint:
        agent.load_models()
        agent.epsilon = agent.epsilon

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
                    time.sleep(0.05)  # Prevents busy-waiting while paused and allows interrupt detection
                
                env.update_fps()
                
              
                
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                if done:
                    reward -= 3

                score += reward
                print(f"current clipped score in episode: {score} | not clipped score: {env.total_score} |  current step in episode: {env.episode_step_counter}")

                if resume_training:
                    agent.store_transition(observation, action, reward, observation_, done)
                    agent.learn()
                if info is not None:
                    time.sleep(1)
                    env.controller.click_(info[0], info[1], clicks=2)
                observation = observation_
                cv2.imshow('Computer Vision', cv2.resize(observation[-1], (320, 320)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    raise KeyboardInterrupt  # Raises KeyboardInterrupt if 'q' is pressed
                
                n_steps += 1

            scores.append(score)
            steps_array.append(n_steps)
            avg_score = np.mean(scores[-50:])
            print(f'episode {i} total_steps {n_steps} score {score:.1f} avg score {avg_score:.1f} best score {best_score:.1f} epsilon {agent.epsilon:.2f}')

            if score > best_score and resume_training:
                agent.save_models()
                best_score = score

            eps_history.append(agent.epsilon)

    except KeyboardInterrupt:
        print("\nExiting program...")
    finally:
        cv2.destroyAllWindows()
        keyboard.unhook_all()
