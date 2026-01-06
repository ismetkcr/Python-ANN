# -*- coding: utf-8 -*-
"""
🚀 ENHANCED DQN with Replay Memory and Target Network
State: 12D vector state (distance, velocity, actions, dx/dy temporal history)
Features: Experience Replay, Target Network, Batch Learning, Real-game Compatible
"""

import numpy as np
import os
import time
from navigation_env import UltraSimpleEnvironment
from agent import Agent
import tensorflow as tf

def manage_memory():
    """Configure TensorFlow memory growth"""
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
    
    # GÖRSELLEŞTIRME KONTROLÜ
    show_game = True  
    
    # TRAINING MODE
    training_mode = True  # True: Train the agent, False: Test only
    
    # Environment setup - Choose training mode:
    
    # MODE 1: Train by PATH NAME (original approach)
    training_path = "anvil_to_armor"  # Specific path training
    env = UltraSimpleEnvironment(render_mode=show_game, path_name=training_path)
    
    
    
    # Available destinations: "anvil", "armor", "inn" (town is start point)
    
    # Training parameters
    best_score = 0  #-np.inf
    load_checkpoint = False
    n_games = 1000  # Back to proper training amount
    
    # Görselleştirme parametreleri
    if show_game:
        visual_delay = 0.02
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
    
    # SIMPLE Agent setup for basic target reaching
    agent = Agent(
        gamma=0.99,               # Standard discount factor
        epsilon=0.015,              # REDUCED initial exploration to prevent excessive rotation
        lr=0.001,                 # High learning rate for simple task
        n_actions=3,              # W, A, D (removed S/backward)
        input_dims=(12,),         # Back to original 12D state vector
        mem_size=20000,           # Larger replay memory for diverse experiences
        batch_size=32,            # Standard batch size
        eps_min=0.05,            # INCREASED min epsilon for continued exploration
        eps_dec=5e-5,             # SLOWER epsilon decay for gradual learning
        replace=500,              # Very frequent target network updates
        algo='DQN',
        env_name='NavigationEnv',
        chkpt_dir='models/'
    )
    
    if load_checkpoint:
        agent.load_models()
        if not training_mode:
            agent.epsilon = 0.01  # Minimal exploration in test mode
    
    # TEST MODE SETUP
    if not training_mode:
        agent.epsilon = 0.01
        show_game = True
        print("🧪 TEST MODE: Agent will use learned policy (epsilon=0.01)")
        print("🎮 Visual mode forced ON for testing")
    
    # Training metrics
    n_steps = 0
    scores, eps_history, steps_array = [], [], []
    success_episodes = []
    success_count = 0
    
   
    
    # Action tracking for debugging
    episode_action_counts = []
    
    print("🚀 Starting Enhanced DQN training with Single-Path Learning...")
    print(f"Environment: {agent.env_name}")
    print(f"Agent: {agent.algo}")
    print(f"Training Mode: {'ON' if training_mode else 'OFF (TEST ONLY)'}")
    print(f"Visual Mode: {'ON' if show_game else 'OFF'}")
    # print(f"🗺️ Path Training: {training_path}")  # For path mode
    #print(f"🎯 Destination Training: {training_destination}")  # For destination mode
    print(f"Current Epsilon: {agent.epsilon:.3f}")
    print(f"Epsilon Decay: {agent.eps_dec:.2e}")
    print(f"Min Epsilon: {agent.eps_min:.3f}")
    print(f"Learning Rate: {agent.lr:.6f}")
    print(f"Batch Size: {agent.batch_size}")
    print(f"Target Update: Every {agent.replace_target_cnt} steps")
    print(f"State Vector: 12D [distance, velocity, actions, dx/dy temporal history]")
    print(f"Real-game Compatible: No agent angle dependency, position-based only")
   
    print("-" * 70)
    
    # Training start time
    start_time = time.time()
    
    try:
        for i in range(n_games):
            done = False
            observation = env.reset()
            score = 0
            episode_steps = 0
            
            # Track actions for this episode
            episode_actions = [0, 0, 0]  # W, A, D counts
            
            # Episode start
            current_success_rate = (success_count / max(1, i)) * 100 if i > 0 else 0
            if show_game:
                env.render()
                time.sleep(episode_start_delay)
            
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                
                score += reward
                episode_steps += 1
                episode_actions[action] += 1
                
                # Store transition in replay memory
                if training_mode:
                    agent.store_transition(observation, action, reward, observation_, done)
                    # Learn from batch (only if enough samples)
                    agent.learn()
                
                observation = observation_
                n_steps += 1
                
                # Render if needed
                if show_game:
                    env.render()
                    if visual_delay > 0:
                        time.sleep(visual_delay)
            
            # Episode statistics
            episode_success = info.get('target_reached', False)
            
            success_episodes.append(episode_success)
            episode_action_counts.append(episode_actions)
            
            if episode_success:
                success_count += 1
            
            scores.append(score)
            steps_array.append(n_steps)
            avg_score = np.mean(scores[-100:])
            success_rate = (success_count / (i + 1)) * 100
            
            # Console output every 10 episodes
            if (i + 1) % 10 == 0:
                elapsed_time = time.time() - start_time
                episodes_per_sec = (i + 1) / elapsed_time
                
                recent_10_successes = sum(success_episodes[-10:])
                recent_10_success_rate = (recent_10_successes / 10) * 100
                
                # Calculate recent action distribution
                recent_actions = episode_action_counts[-10:]
                total_recent_actions = [sum(ep[j] for ep in recent_actions) for j in range(3)]
                total_count = sum(total_recent_actions)
                recent_action_pcts = [a/total_count * 100 for a in total_recent_actions] if total_count > 0 else [0,0,0]
                recent_rotation_pct = (total_recent_actions[1] + total_recent_actions[2]) / total_count * 100 if total_count > 0 else 0
                
                mode_indicator = "TEST" if not training_mode else "TRAIN"
                
                # Get curriculum info if available
                curriculum_info = f" [L{info.get('curriculum_level', 0)}:{info.get('level_success_rate', 0)*100:.1f}%]"
                
                print(f'[{mode_indicator}] episode {i+1:4d}/{n_games:4d} score {score:8.1f} avg score {avg_score:8.1f} '
                      f'best score {best_score:8.1f} epsilon {agent.epsilon:.3f} steps {episode_steps:3d} '
                      f'success rate {success_rate:.1f}% time {elapsed_time:.1f}s speed {episodes_per_sec:.1f}eps/s{curriculum_info}')
                print(f'   Recent 10: success {recent_10_success_rate:.1f}% | Actions: W:{recent_action_pcts[0]:.1f}% A:{recent_action_pcts[1]:.1f}% D:{recent_action_pcts[2]:.1f}% | Rotation:{recent_rotation_pct:.1f}%')
            
            # Additional detailed stats every 100 episodes
            if (i + 1) % 100 == 0:
                agent.save_models()
                recent_100_successes = sum(success_episodes[-100:]) if len(success_episodes) >= 100 else sum(success_episodes)
                recent_100_success_rate = (recent_100_successes / min(100, len(success_episodes))) * 100
                
                print(f"\n{'='*70}")
                print(f"EPISODE {i+1} - 100 EPISODE SUMMARY")
                print(f"{'='*70}")
                print(f"Recent 100 Episodes Success Rate: {recent_100_success_rate:.1f}%")
                print(f"Overall Success Rate: {success_rate:.1f}%")
                print(f"Average Score (last 100): {avg_score:.1f}")
                print(f"Best Score: {best_score:.1f}")
                print(f"Epsilon: {agent.epsilon:.3f}")
                print(f"Learning Steps: {agent.learn_step_counter}")
                print(f"Total Steps: {n_steps}")
                print(f"Training Time: {elapsed_time/60:.1f} minutes")
                
                # SINGLE PATH PERFORMANCE
                print(f"\n📊 SINGLE-PATH PERFORMANCE (town_to_anvil):")
                print(f"Episodes completed: {i+1}")
                print(f"Successful episodes: {success_count}")
                print(f"Current success rate: {success_rate:.1f}%")
                
                print(f"{'='*70}")
                
            
            # Save best model
            if training_mode and score > best_score:
                agent.save_models()
                best_score = score
                if (i + 1) % 10 == 0:  # Only print if it's a 10-episode mark
                    print(f"🏆 New best score! Model saved at episode {i+1}")
            
            eps_history.append(agent.epsilon)
            
            # Episode delay
            if show_game:
                if episode_success:
                    time.sleep(success_delay)
                else:
                    time.sleep(fail_delay)
    
    except KeyboardInterrupt:
        print("\n⏹️ Training interrupted by user")
    
    finally:
        # Final save
        if training_mode:
            agent.save_models()
            print("💾 Final model saved!")
        else:
            print("🧪 Test completed - no model saving in test mode")
        
        # Training time
        total_time = time.time() - start_time
        
        # Print final statistics
        mode_text = "TRAINING" if training_mode else "TESTING"
        print("\n" + "=" * 80)
        print(f"🚀 ENHANCED DQN {mode_text} COMPLETED!")
        print("=" * 80)
        print(f"{'FINAL RESULTS':<30} {'VALUE':<20}")
        print("-" * 50)
        print(f"{'Total Episodes':<30} {len(scores):<20}")
        print(f"{'Total Time':<30} {total_time:.1f} seconds")
        print(f"{'Average Time per Episode':<30} {total_time/len(scores):.2f} seconds" if len(scores) > 0 else "")
        print(f"{'Episodes per Second':<30} {len(scores)/total_time:.1f}" if len(scores) > 0 else "")
        print(f"{'Total Steps':<30} {n_steps:<20}")
        print(f"{'Learn Steps':<30} {agent.learn_step_counter:<20}")
        print(f"{'Best Score':<30} {best_score:.1f}")
        print(f"{'Final Average Score':<30} {np.mean(scores[-100:]):.1f}" if len(scores) >= 100 else f"{'Final Average Score':<30} {np.mean(scores):.1f}" if len(scores) > 0 else "")
        print(f"{'Final Epsilon':<30} {agent.epsilon:.3f}")
        
        if len(scores) > 0:
            print()
            print("SUCCESS STATISTICS:")
            print("-" * 50)
            final_success_rate = (success_count / len(scores)) * 100
            print(f"{'Total Successful Episodes':<30} {success_count}")
            print(f"{'Overall Success Rate':<30} {final_success_rate:.1f}%")
            
            if len(success_episodes) >= 100:
                recent_successes = sum(success_episodes[-100:])
                recent_success_rate = (recent_successes / 100) * 100
                print(f"{'Final Success Rate (100)':<30} {recent_success_rate:.1f}%")
            
            # Final action distribution
            if episode_action_counts:
                all_actions = [sum(ep[j] for ep in episode_action_counts) for j in range(3)]
                total_actions = sum(all_actions)
                if total_actions > 0:
                    print()
                    print("OVERALL ACTION DISTRIBUTION:")
                    print("-" * 50)
                    print(f"{'Forward (W)':<30} {all_actions[0]/total_actions*100:5.1f}%")
                    print(f"{'Left (A)':<30} {all_actions[1]/total_actions*100:5.1f}%")
                    print(f"{'Right (D)':<30} {all_actions[2]/total_actions*100:5.1f}%")
                    print(f"{'Total Rotation':<30} {(all_actions[1]+all_actions[2])/total_actions*100:5.1f}%")
            
            # FINAL SINGLE-PATH PERFORMANCE SUMMARY
            print()
            print("FINAL SINGLE-PATH PERFORMANCE (town_to_anvil):")
            print("-" * 50)
            print(f"{'Total Episodes':<20} {len(scores)}")
            print(f"{'Successful Episodes':<20} {success_count}")
            print(f"{'Final Success Rate':<20} {final_success_rate:.1f}%")
            
            if final_success_rate >= 70:
                print(f"\n✅ Path well-learned: town_to_anvil ({final_success_rate:.1f}%)")
            elif final_success_rate >= 50:
                print(f"\n🟡 Path partially learned: town_to_anvil ({final_success_rate:.1f}%)")
            else:
                print(f"\n❌ Path needs more training: town_to_anvil ({final_success_rate:.1f}%)")
                print("💡 Consider increasing training episodes")
        
        print("=" * 80)
        
        # Close environment
        env.close()
        
        print(f"\n🚀 Enhanced DQN {mode_text.lower()} completed!")