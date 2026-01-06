# -*- coding: utf-8 -*-
"""
🔧 FIXED Enhanced Simple DQN Training Script
State: 12D enhanced state with action history and movement tracking
Reward: HEDEFE ODAKLI reward system
Success Rate: SADECE hedefe ulaşılan episodlar
"""

import numpy as np
import os
import time
import matplotlib.pyplot as plt
from navigation_env import VisualNavigationEnvironment
import tensorflow.keras as keras
import tensorflow as tf

def create_dqn_network(lr, n_actions):
    """Create DQN network using Sequential API - much cleaner!"""
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(12,)),  # ⭐ 12D state input
        keras.layers.Dense(16, activation='relu'),                      # ⭐ Second layer
        keras.layers.Dense(n_actions)                                   # Output layer
    ])
    
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr))
    return model

class Agent:
    def __init__(self, n_actions, lr, gamma=0.99,
                 epsilon=1, eps_dec=1e-5, eps_min=0.01,
                 chkpt_dir='models/', algo='FixedEnhancedDQN', env_name='Navigation'):
        self.lr = lr
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]
        
        # Model saving parameters
        self.chkpt_dir = chkpt_dir
        self.algo = algo
        self.env_name = env_name
        self.fname = self.chkpt_dir + self.env_name + '_' + self.algo + '_'
        
        # Initialize Q network using Sequential API
        self.Q = create_dqn_network(self.lr, n_actions)
    
    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            actions = self.Q(state)
            
            action = tf.argmax(actions, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min
           
    def learn(self, state, action, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        next_state = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor([reward], dtype=tf.float32)
        action = tf.convert_to_tensor([action], dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            q_pred = self.Q(state)[0]  # predicted Q val
            q_pred = q_pred[int(action)]
            
            if done:
                q_target = reward  # No future reward if episode ended
            else:
                q_next = tf.reduce_max(self.Q(next_state), axis=1)
                q_target = reward + self.gamma * q_next
            
            loss = tf.keras.losses.MSE(q_target, q_pred)
            
        params = self.Q.trainable_variables
        grads = tape.gradient(loss, params)
        self.Q.optimizer.apply_gradients(zip(grads, params))
        self.decrement_epsilon()
    
    def save_models(self):
        """Save the Q network in H5 format"""
        model_path = self.fname + 'q_network.h5'
        self.Q.save(model_path)
        print(f'💾 Model saved successfully as: {model_path}')
    
    def load_agent_for_train(self):
        """Load the Q network for training/continue training"""
        model_path = r"C:\Users\ismt\Desktop\python-ann\KnightOnlineFarmAgentProject\ko_upgrade\RL_agent\models\VisualNavigationEnvironment_FixedEnhancedDQN_q_network.h5"
        try:
            self.Q = keras.models.load_model(model_path, compile=False)
            self.Q.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr))
            print('✅ Training Model loaded successfully')
            print(f'📁 Path: {model_path}')
        except Exception as e:
            print(f'❌ CRITICAL ERROR: Cannot load training model!')
            print(f'❌ Error: {e}')
            print(f'❌ Path: {model_path}')
            print('🛑 Program stopped - Model file is required!')
            import sys
            sys.exit(1)
    
    def load_agent_for_test(self):
        """Load the Q network from specific trained model path"""
        trained_model_path = r"C:\Users\ismt\Desktop\python-ann\KnightOnlineFarmAgentProject\ko_upgrade\RL_agent\models\TRAINED_FixedEnhancedDQN_q_network/trained_model.h5"
        try:
            self.Q = keras.models.load_model(trained_model_path, compile=False)
            self.Q.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr))
            print('✅ TRAINED Model loaded successfully for testing')
            print(f'📁 Path: {trained_model_path}')
            
        except Exception as e:
            print(f'❌ CRITICAL ERROR: Cannot load trained model!')
            print(f'❌ Error: {e}')
            print(f'❌ Path: {trained_model_path}')
            print('🛑 Program stopped - Trained model file is required for testing!')
            import sys
            sys.exit(1)
           

def manage_memory():
    """Simple memory management for TensorFlow"""
    import tensorflow as tf
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Memory growth setting error: {e}")

def create_directories():
    """Create necessary directories"""
    os.makedirs('models/', exist_ok=True)
    os.makedirs('plots/', exist_ok=True)

if __name__ == '__main__':
    # Setup
    manage_memory()
    create_directories()
    
    # 🎮 GÖRSELLEŞTIRME KONTROLÜ
    show_game = False  # True: Oyun penceresi açılır, False: Sadece konsol çıktısı
    
    # 🧪 TRAINING MODE - NEW ADDITION
    training_mode = True  # True: Train the agent, False: Test only (no learning)
    
    # Environment setup
    env = VisualNavigationEnvironment(render_mode=show_game)
    
    # Training parameters
    best_score = 0
    load_checkpoint = False
    n_games = 500
    
    # Görselleştirme parametreleri
    if show_game:
        visual_delay = 0.05
        episode_start_delay = 0.5
        success_delay = 1.0
        fail_delay = 0.3
        print("🎮 VISUAL MODE: ON - Game window will be displayed")
    else:
        visual_delay = 0.0
        episode_start_delay = 0.0
        success_delay = 0.0
        fail_delay = 0.0
        print("⚡ FAST MODE: ON - No game window, faster training")
    
    # 🔧 FIXED Agent setup - Simple DQN with better hyperparameters
    agent = Agent(
        n_actions=4,              # W, A, S, D
        lr=0.001,                 # Learning rate
        gamma=0.99,               # Discount factor
        epsilon=1,              # Initial exploration rate
        eps_dec=5e-5,             # ⭐ SLOWER epsilon decay (was 1e-5, now 2e-6)
        eps_min=0.08,             # ⭐ HIGHER minimum epsilon (was 0.01, now 0.08)
        chkpt_dir='models/',
        algo='FixedEnhancedDQN',
        env_name='VisualNavigationEnvironment'
    )
    
    if load_checkpoint:
        agent.load_agent_for_train()
        #agent.load_agent_for_test()
    
        
    
    # 🧪 TEST MODE SETUP - NEW ADDITION
    if not training_mode and load_checkpoint:
        agent.epsilon = 0.02  # No exploration in test mode
        show_game = True     # Force visualization in test mode
        print("🧪 TEST MODE: Agent will use learned policy only (epsilon=0)")
        print("🎮 Visual mode forced ON for testing")
    
    # File naming
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    
    # Training loop
    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    
    # 🎯 SUCCESS TRACKING - Sadece hedefe ulaşma odaklı
    success_episodes = []  # Her episode için True/False listesi
    success_count = 0      # Toplam başarılı episode sayısı
    
    print("🔧 Starting HEDEFE ODAKLI Enhanced DQN training...")
    print(f"Environment: {agent.env_name}")
    print(f"Agent: {agent.algo}")
    print(f"Training Mode: {'ON' if training_mode else 'OFF (TEST ONLY)'}")
    print(f"Visual Mode: {'ON' if show_game else 'OFF'}")
    print(f"Current Epsilon: {agent.epsilon:.3f}")
    print(f"Epsilon Decay: {agent.eps_dec:.2e} (SLOWER)")
    print(f"Min Epsilon: {agent.eps_min:.3f} (HIGHER)")
    print(f"Current Learning Rate: {agent.lr:.6f}")
    print("🎯 HEDEFE ODAKLI REWARD SYSTEM:")
    print("   • SUCCESS: 100-120 (efficiency bonus)")
    print("   • Progress: +8x yaklaşma, -12x uzaklaşma")
    print("   • Collision: -15, Timeout: -25, Standing: -2.0")
    print("   • Time Penalty: -0.05/step")
    print("   • ❌ Proximity bonus YOK - Sadece hedefe ulaşma!")
    print("-" * 70)
    
    # Training başlangıç zamanı
    start_time = time.time()
    
    try:
        for i in range(n_games):
            done = False
            observation = env.reset()
            score = 0
            episode_steps = 0
            
            # Episode başlangıcı
            current_success_rate = (success_count / max(1, i)) * 100 if i > 0 else 0
            if show_game:
                env.render(episode=i+1, score=score, epsilon=agent.epsilon, 
                          success_rate=current_success_rate)
                time.sleep(episode_start_delay)
            
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                
                score += reward
                episode_steps += 1
                
                # 🎯 FIXED DQN LEARNING - Her step'te öğren (ONLY IF TRAINING)
                if training_mode:
                    try:
                        agent.learn(observation, action, reward, observation_, done)
                    except Exception as e:
                        if show_game:
                            print(f"⚠️ Learning error: {e}")
                
                observation = observation_
                n_steps += 1
                
                # Görselleştirme
                if show_game:
                    env.render(episode=i+1, score=score, epsilon=agent.epsilon,
                              success_rate=current_success_rate)
                    if visual_delay > 0:
                        time.sleep(visual_delay)
                
                # Break if episode is too long
                if episode_steps > 1000:
                    if show_game:
                        print(f"Episode {i} too long, breaking...")
                    break
            
            # 🎯 SUCCESS TRACKING - SADECE hedefe ulaşma
            episode_success = info['target_reached']
            success_episodes.append(episode_success)
            
            if episode_success:
                success_count += 1
                if show_game:
                    print(f"🎯 Episode {i+1}: SUCCESS! Target reached in {episode_steps} steps (Score: {score:.1f})")
            else:
                if show_game:
                    print(f"❌ Episode {i+1}: Failed after {episode_steps} steps (Score: {score:.1f})")
            
            scores.append(score)
            steps_array.append(n_steps)
            avg_score = np.mean(scores[-100:])
            success_rate = (success_count / (i + 1)) * 100
            
            # Konsol çıktısı
            if show_game or (i + 1) % 5 == 0:
                mode_indicator = "TEST" if not training_mode else "TRAIN"
                print('[{}] episode {:4d}/{:4d} score {:8.1f} avg score {:8.1f} '
                      'best score {:8.1f} epsilon {:.3f} steps {:3d} '
                      'success rate {:.1f}%'.format(
                          mode_indicator, i+1, n_games, score, avg_score, best_score, agent.epsilon,
                          episode_steps, success_rate))
            
            # Save best model (ONLY IF TRAINING)
            if training_mode and score > best_score:
                agent.save_models()
                if show_game:
                    print(f"🏆 New best score! Model saved at episode {i+1}")
                best_score = score
            
            eps_history.append(agent.epsilon)
            
            # 🎯 ENHANCED Progress raporu - Her 50 episode'da
            if (i + 1) % 50 == 0:
                elapsed_time = time.time() - start_time
                episodes_per_sec = (i + 1) / elapsed_time
                
                # Son 50 episode success rate hesaplama
                recent_50_successes = sum(success_episodes[-50:])
                recent_50_success_rate = (recent_50_successes / 50) * 100
                
                mode_text = "TRAINING" if training_mode else "TESTING"
                print("\n" + "=" * 80)
                print(f"🎯 HEDEFE ODAKLI PROGRESS REPORT - Episode {i+1} ({mode_text})")
                print("=" * 80)
                print(f"🔄 Status:")
                print(f"   Mode: {mode_text}")
                print(f"   Current Epsilon: {agent.epsilon:.3f}")
                print(f"   Learning Rate: {agent.lr:.6f}")
                print(f"   Best Score: {best_score:.1f}")
                print(f"   Average Score (last 50): {np.mean(scores[-50:]):.1f}")
                print()
                print(f"🎯 SUCCESS STATISTICS (Target Reached Only):")
                print(f"   Overall Success Rate: {success_rate:.1f}% ({success_count}/{i+1} episodes)")
                print(f"   Recent Success Rate (last 50): {recent_50_success_rate:.1f}% ({recent_50_successes}/50 episodes)")
                print(f"   Total Successful Episodes: {success_count}")
                print()
                print(f"⚡ Performance:")
                print(f"   Speed: {episodes_per_sec:.1f} episodes/sec")
                print(f"   Elapsed Time: {elapsed_time:.1f} seconds")
                print(f"   Average Time per Episode: {elapsed_time/(i+1):.2f} seconds")
                print()
                print(f"🏆 Achievement Progress:")
                if success_rate >= 80:
                    print(f"   🌟 EXCELLENT! {success_rate:.1f}% success rate")
                elif success_rate >= 60:
                    print(f"   🔥 GOOD! {success_rate:.1f}% success rate")  
                elif success_rate >= 40:
                    print(f"   📈 IMPROVING! {success_rate:.1f}% success rate")
                elif success_rate >= 20:
                    print(f"   🌱 LEARNING! {success_rate:.1f}% success rate")
                else:
                    print(f"   🔄 EXPLORING! {success_rate:.1f}% success rate")
                print("=" * 80 + "\n")
            
            # Episode arası bekletme
            if show_game:
                if episode_success:
                    time.sleep(success_delay)
                else:
                    time.sleep(fail_delay)
    
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    
    finally:
        # Final save (ONLY IF TRAINING)
        if training_mode:
            agent.save_models()
            print("💾 Final model saved!")
        else:
            print("🧪 Test completed - no model saving in test mode")
        
        # Training süresi hesaplama
        total_time = time.time() - start_time
        
        # Print final statistics
        mode_text = "TRAINING" if training_mode else "TESTING"
        print("\n" + "=" * 80)
        print(f"🎯 HEDEFE ODAKLI DQN {mode_text} COMPLETED!")
        print("=" * 80)
        print(f"Total episodes: {len(scores)}")
        print(f"Total time: {total_time:.1f} seconds")
        print(f"Mode: {mode_text}")
        
        if len(scores) > 0:
            print(f"Average time per episode: {total_time/len(scores):.2f} seconds")
            print(f"Speed: {len(scores)/total_time:.1f} episodes/sec")
            print(f"Best score: {best_score:.1f}")
            if len(scores) >= 100:
                print(f"Final average score (last 100): {np.mean(scores[-100:]):.1f}")
            else:
                print(f"Average score: {np.mean(scores):.1f}")
            
            print()
            print(f"🎯 SUCCESS STATISTICS (Target Reached Only):")
            final_success_rate = (success_count / len(scores)) * 100
            print(f"   Total successful episodes: {success_count}")
            print(f"   Overall success rate: {final_success_rate:.1f}%")
            
            # Son 100 episode success rate
            if len(success_episodes) >= 100:
                recent_successes = sum(success_episodes[-100:])
                recent_success_rate = (recent_successes / 100) * 100
                print(f"   Final success rate (last 100): {recent_success_rate:.1f}%")
            
        else:
            print("No episodes completed successfully")
            print(f"Stopped early - check for errors")
        
        print()
        print(f"Final epsilon: {agent.epsilon:.3f}")
        print(f"Learning rate: {agent.lr:.6f}")
        print(f"Total steps: {n_steps}")
        print(f"Visual mode: {'ON' if show_game else 'OFF'}")
        
        print("\n🎯 HEDEFE ODAKLI ENHANCEMENTS:")
        print("   • 12D Enhanced State Vector")
        print("   • Action History Tracking (3 steps)")
        print("   • Multi-step Movement Tracking") 
        print("   • Target Direction Information")
        print("   • HEDEFE ODAKLI Reward System")
        print("   • SUCCESS = Target Reached ONLY")
        print("   • Agresif uzaklaşma cezası (-12x)")
        print("   • Sürekli time pressure (-0.05/step)")
        
        # 🎯 FIXED Create enhanced plots - SUCCESS RATE DÜZELTME
        try:
            plt.figure(figsize=(16, 12))
            
            # Plot 1: Scores
            plt.subplot(2, 3, 1)
            plt.plot(scores, alpha=0.6, color='blue', label='Episode Score')
            if len(scores) > 10:
                moving_avg = [np.mean(scores[max(0, i-10):i+1]) for i in range(len(scores))]
                plt.plot(moving_avg, color='red', linewidth=2, label='Moving Average (10)')
            title_suffix = " (TEST)" if not training_mode else " (TRAINING)"
            plt.title('Episode Scores (HEDEFE ODAKLI)' + title_suffix)
            plt.xlabel('Episode')
            plt.ylabel('Score')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Plot 2: Epsilon decay
            plt.subplot(2, 3, 2)
            plt.plot(eps_history, color='orange')
            epsilon_title = 'Epsilon (Fixed at 0 - TEST)' if not training_mode else 'Epsilon Decay (Slower)'
            plt.title(epsilon_title)
            plt.xlabel('Episode')
            plt.ylabel('Epsilon')
            plt.grid(True, alpha=0.3)
            
            # Plot 3: Steps per episode
            plt.subplot(2, 3, 3)
            episode_steps_list = [steps_array[i] - (steps_array[i-1] if i > 0 else 0) 
                            for i in range(len(steps_array))]
            plt.plot(episode_steps_list, alpha=0.6, color='green')
            plt.title('Steps per Episode')
            plt.xlabel('Episode')
            plt.ylabel('Steps')
            plt.grid(True, alpha=0.3)
            
            # 🎯 FIXED Plot 4: GERÇEK Success rate (sadece target_reached)
            plt.subplot(2, 3, 4)
            window_size = min(20, len(success_episodes))
            success_rates = []
            for i in range(len(success_episodes)):
                start_idx = max(0, i - window_size + 1)
                window_successes = success_episodes[start_idx:i+1]
                success_rate_window = sum(window_successes) / len(window_successes) * 100
                success_rates.append(success_rate_window)
            
            plt.plot(success_rates, color='purple')
            success_title_suffix = " (TEST)" if not training_mode else " (TRAINING)"
            plt.title(f'GERÇEK Success Rate (Target Reached, Rolling {window_size})' + success_title_suffix)
            plt.xlabel('Episode')
            plt.ylabel('Success Rate (%)')
            plt.grid(True, alpha=0.3)
            
            # Plot 5: Score distribution
            plt.subplot(2, 3, 5)
            plt.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            dist_title_suffix = " (TEST)" if not training_mode else " (TRAINING)"
            plt.title('Score Distribution (HEDEFE ODAKLI)' + dist_title_suffix)
            plt.xlabel('Score')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            
            # Plot 6: Training progress trend
            plt.subplot(2, 3, 6)
            if len(scores) > 50:
                chunk_size = max(1, len(scores) // 10)
                chunk_avgs = []
                for i in range(0, len(scores), chunk_size):
                    chunk = scores[i:i+chunk_size]
                    chunk_avgs.append(np.mean(chunk))
                plt.plot(range(0, len(scores), chunk_size), chunk_avgs, 
                        marker='o', linewidth=2, color='red')
                progress_title = 'Performance Progress' if not training_mode else 'Learning Progress (Chunked Averages)'
                plt.title(progress_title)
                plt.xlabel('Episode')
                plt.ylabel('Average Score')
                plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(figure_file, dpi=150, bbox_inches='tight')
            print(f"📈 HEDEFE ODAKLI plots saved as {figure_file}")
            
        except ImportError:
            print("Matplotlib not available. Plots not generated.")
        except Exception as e:
            print(f"Error creating plots: {e}")
        
        # Close pygame window
        env.close()
        
        mode_text = "Test" if not training_mode else "Training"
        print(f"\n🎯 HEDEFE ODAKLI Enhanced DQN {mode_text.lower()} completed! (Visual mode: {'ON' if show_game else 'OFF'})")
        print("📝 Note: SUCCESS = Target Reached ONLY (info['target_reached'])")
        if not training_mode:
            print("🧪 TEST MODE: Agent used learned policy only (no exploration/learning)")
        else:
            print("🚀 Expected: Agent focuses on reaching target, not collecting points!")