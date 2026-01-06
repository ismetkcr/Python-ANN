import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from network import DeepQNetwork
from replay_memory import ReplayBuffer


class Agent:
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                 mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                 replace=1000, algo=None, env_name=None, chkpt_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.fname = self.chkpt_dir + self.env_name + '_' + self.algo + '_'
        self.mem_size=mem_size

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)

        self.q_eval = DeepQNetwork(input_dims, n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr))
        self.q_next = DeepQNetwork(input_dims, n_actions)
        self.q_next.compile(optimizer=Adam(learning_rate=lr))

    def save_models(self):
        self.q_eval.save(self.fname+'q_eval.h5')
        self.q_next.save(self.fname+'q_next.h5')
        print('... models saved successfully ...')

    def load_models(self, path_name=None):
        """Load models with compatibility fixes for newer TensorFlow versions"""
        try:
            # Try loading with custom objects for compatibility
            import tensorflow as tf
            from tensorflow import keras
            import os
            
            print("Loading models...")
            # Use current directory relative path instead of hardcoded Windows path
            self.real_model_path = os.path.join(os.path.dirname(__file__), "models_real")
            
            # Default to armor_to_anvil if no path specified
            if path_name is None:
                path_name = "armor_to_anvil"
            
            # Custom objects for compatibility
            custom_objects = {}
            
            # Load models with error handling
            try:
                q_eval_path = os.path.join(self.real_model_path, f'q_eval_trained_{path_name}.h5')
                self.q_eval = keras.models.load_model(
                    q_eval_path,
                    custom_objects=custom_objects,
                    compile=False  # Don't compile initially
                )
                print(f"✅ q_eval model loaded successfully for {path_name}")
            except Exception as e:
                print(f"❌ Error loading q_eval for {path_name}: {e}")
                raise
                
            try:
                q_next_path = os.path.join(self.real_model_path, f'q_next_trained_{path_name}.h5')
                self.q_next = keras.models.load_model(
                    q_next_path,
                    custom_objects=custom_objects,
                    compile=False  # Don't compile initially
                )
                print(f"✅ q_next model loaded successfully for {path_name}")
            except Exception as e:
                print(f"❌ Error loading q_next for {path_name}: {e}")
                raise
            
            # Recompile models with current optimizer settings
            self.q_eval.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                loss='mse'
            )
            self.q_next.compile(
                optimizer=keras.optimizers.Adam(learning_rate=self.lr),
                loss='mse'
            )
            
            # Store the loaded path for reference
            self.loaded_path = path_name
            
            print("✅ Models recompiled with current optimizer settings")
            print(f"✅ Learning rate set to: {self.lr}")
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            print("💡 TROUBLESHOOTING TIPS:")
            print("   1. Check if model files exist in 'models_real/' directory")
            print("   2. Try recreating models with current TensorFlow version")
            print("   3. Check file permissions")
            print("   4. Verify model file integrity")
            raise RuntimeError(f"Model loading failed: {e}")
            

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                                  self.memory.sample_buffer(self.batch_size)
        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        states_ = tf.convert_to_tensor(new_state)
        return states, actions, rewards, states_, dones

    def choose_action(self, observation): #convert to tensor denenecek... expand dims yerine..
        if np.random.random() > self.epsilon:
            observation = np.array(observation, dtype=np.float32)
            #state = np.expand_dims(observation, axis=0)
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            actions = self.q_eval(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        return action

    def replace_target_network(self):
        self.q_next.set_weights(self.q_eval.get_weights()) 

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
                           if self.epsilon > self.eps_min else self.eps_min

    def learn(self):
        """Learn only if enough experiences are stored"""
        # Yeterli experience var mı kontrol et
        if self.memory.mem_cntr < self.batch_size:
            return  # Yeterli experience yok, öğrenme yapma
        
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.replace_target_network()
            
        # Sample a batch of experiences from memory
        # states.shape = (32, 5)              -> (32, 5) for 5-dimensional state vector
        # actions.shape = (32,)               -> (32,) because we have one action per state in the batch
        # rewards.shape = (32,)               -> (32,) because we have one reward per state in the batch
        # states_.shape = (32, 5)             -> Same as `states`, the next state
        # dones.shape = (32,)                 -> (32,), indicating whether the episode ended after the action
        try:
            states, actions, rewards, states_, dones = self.sample_memory()
        except ValueError as e:
            print(f"⚠️ Memory sampling error: {e}")
            print(f"Memory count: {self.memory.mem_cntr}, Batch size: {self.batch_size}")
            return
        
        # indices.shape = (32,)               -> A tensor of indices [0, 1, 2, ..., 31]
        indices = tf.range(self.batch_size, dtype=tf.int32)
        # action_indices.shape = (32, 2)      -> Stack of indices and actions
        action_indices = tf.stack([indices, actions], axis=1)
        
        with tf.GradientTape() as tape:
            # q_eval(states).shape = (32, 4)   -> Network output, Q-values for 4 actions (W,A,S,D)
            # q_pred.shape = (32,)             -> Gather Q-values for the specific actions taken
            q_pred = tf.gather_nd(self.q_eval(states), indices=action_indices)
            # q_next.shape = (32, 4)           -> Network output for next states
            q_next = self.q_next(states_)
            # max_actions.shape = (32,)        -> The index of the action with the highest Q-value
            max_actions = tf.math.argmax(q_next, axis=1, output_type=tf.int32)
            # max_action_idx.shape = (32, 2)   -> Stack of indices and max_actions
            max_action_idx = tf.stack([indices, max_actions], axis=1)
            # q_target.shape = (32,)           -> Target Q-value for each state-action pair
            q_target = rewards + \
                self.gamma*tf.gather_nd(q_next, indices=max_action_idx) *\
                (1 - tf.cast(dones, tf.float32))
            # loss.shape = ()                  -> The loss is scalar
            loss = keras.losses.MSE(q_pred, q_target)
        
        # Apply gradients
        params = self.q_eval.trainable_variables
        grads = tape.gradient(loss, params)
        self.q_eval.optimizer.apply_gradients(zip(grads, params))
        self.learn_step_counter += 1
        self.decrement_epsilon()
    
    def get_available_paths(self):
        """
        📍 REAL GAME PATH DEFINITIONS - For use in RL_in_real_game/rl_test.py
        
        USAGE IN REAL GAME:
        -------------------
        # In RL_in_real_game/rl_test.py:
        agent = Agent(...)
        agent.load_models()
        paths = agent.get_available_paths()
        
        # Select path for navigation
        target_path = "town_to_anvil" 
        path_coords = agent.get_path_coordinates(target_path)
        # Use path_coords with real game navigation logic
        
        AVAILABLE PATHS:
        ----------------
        - "town_to_anvil"  : Town → Blacksmith
        - "anvil_to_armor" : Blacksmith → Armor Shop  
        - "armor_to_inn"   : Armor Shop → Inn
        - "inn_to_armor"   : Inn → Armor Shop
        - "armor_to_anvil" : Armor Shop → Blacksmith
        
        RETURNS:
        --------
        list: All available path names for real game navigation
        """
        return [
            "town_to_anvil", "anvil_to_armor", "armor_to_inn", 
            "inn_to_armor", "armor_to_anvil"
        ]
    
    def get_path_coordinates(self, path_name):
        """
        🗺️ GET PATH COORDINATES - Returns actual game coordinates for a path
        
        ⚠️ DEPRECATED: Use MultiAgentNavigationSystem directly for better control
        
        PREFERRED USAGE:
        ----------------
        from multi_agent_test import MultiAgentNavigationSystem
        nav_system = MultiAgentNavigationSystem()
        path_info = nav_system.available_agents["town_to_anvil"]
        start_pos = path_info["start"]  # (822, 533)
        end_pos = path_info["end"]      # (814, 604)
        
        PARAMETERS:
        -----------
        path_name (str): Name of the path
        
        RETURNS:
        --------
        list: [start_pos, end_pos] from multi_agent_test.py
        """
        # PATH COORDINATES MOVED TO MULTI_AGENT_TEST.PY
        # This method now gets coordinates from the centralized location
        try:
            from multi_agent_test import MultiAgentNavigationSystem
            nav_system = MultiAgentNavigationSystem()
            
            if path_name not in nav_system.available_agents:
                available = list(nav_system.available_agents.keys())
                raise ValueError(f"Unknown path '{path_name}'. Available: {available}")
            
            path_info = nav_system.available_agents[path_name]
            # Return simple start/end format for compatibility
            return [path_info["start"], path_info["end"]]
            
        except ImportError:
            print("❌ Could not import MultiAgentNavigationSystem")
            print("💡 Use MultiAgentNavigationSystem directly for path coordinates")
            raise ValueError("Path coordinates moved to multi_agent_test.py")
    
    def get_destination_coordinates(self, destination):
        """
        🎯 GET DESTINATION COORDINATES - Returns target coordinates for real game
        
        ⚠️ DEPRECATED: Use MultiAgentNavigationSystem directly for better control
        
        PREFERRED USAGE:
        ----------------
        from multi_agent_test import MultiAgentNavigationSystem
        nav_system = MultiAgentNavigationSystem()
        # Get anvil coordinates:
        anvil_coords = nav_system.available_agents["town_to_anvil"]["end"]  # (814, 604)
        
        AVAILABLE DESTINATIONS (from multi_agent_test.py):
        ---------------------------------------------------
        - "anvil": town_to_anvil["end"] coordinates
        - "armor": anvil_to_armor["end"] coordinates  
        - "inn": armor_to_inn["end"] coordinates
        
        PARAMETERS:
        -----------
        destination (str): Where to go ("anvil", "armor", "inn")
        
        RETURNS:
        --------
        tuple: (x, y) coordinates from multi_agent_test.py
        """
        # DESTINATION COORDINATES NOW DERIVED FROM MULTI_AGENT_TEST.PY
        # Get coordinates from the centralized path definitions
        try:
            from multi_agent_test import MultiAgentNavigationSystem
            nav_system = MultiAgentNavigationSystem()
            
            # Map destination names to actual coordinates from available_agents
            destination_map = {
                "anvil": nav_system.available_agents["town_to_anvil"]["end"],     # (814, 604)
                "armor": nav_system.available_agents["anvil_to_armor"]["end"],    # (767, 599)  
                "inn": nav_system.available_agents["armor_to_inn"]["end"]         # (764, 648)
            }
            
            if destination not in destination_map:
                available = list(destination_map.keys())
                raise ValueError(f"Unknown destination '{destination}'. Available: {available}")
            
            return destination_map[destination]
            
        except ImportError:
            print("❌ Could not import MultiAgentNavigationSystem")
            print("💡 Use MultiAgentNavigationSystem directly for destination coordinates")
            raise ValueError("Destination coordinates moved to multi_agent_test.py")
    
    def get_available_destinations(self):
        """
        📍 GET AVAILABLE DESTINATIONS - For real game usage
        
        REAL GAME USAGE:
        ----------------
        # In RL_in_real_game/rl_test.py:
        destinations = agent.get_available_destinations()
        # Returns: ["anvil", "armor", "inn"]
        
        # Let player choose destination:
        for dest in destinations:
            print(f"Go to {dest}: agent.set_target_destination('{dest}')")
        
        RETURNS:
        --------
        list: All available destination names (town is start point, not destination)
        """
        return ["anvil", "armor", "inn"]
    
    def set_target_destination(self, destination):
        """
        🎯 SET TARGET DESTINATION - Tell agent where to navigate in real game
        
        REAL GAME USAGE:
        ----------------
        # In RL_in_real_game/rl_test.py:
        agent.set_target_destination("anvil")
        
        # Now in game loop:
        while not_reached_destination():
            action = agent.choose_action(current_state)
            execute_action_in_game(action)
        
        PARAMETERS:
        -----------
        destination (str): Where to go ("town", "anvil", "armor", "inn")
        
        EFFECT:
        -------
        Sets internal target coordinates that agent will navigate towards
        """
        target_coords = self.get_destination_coordinates(destination)
        self.current_target = target_coords
        self.current_destination = destination
        
        print(f"🎯 Agent target set to: {destination} at {target_coords}")
    
    def get_current_target(self):
        """
        📍 GET CURRENT TARGET - Returns where agent is currently trying to go
        
        REAL GAME USAGE:
        ----------------
        # Check where agent is heading
        if hasattr(agent, 'current_target'):
            target = agent.get_current_target()
            print(f"Agent heading to: {target}")
        
        RETURNS:
        --------
        dict: Current target info or None if no target set
        """
        if hasattr(self, 'current_target') and hasattr(self, 'current_destination'):
            return {
                'destination': self.current_destination,
                'coordinates': self.current_target
            }
        return None
    
    def go_target(self, game_controller, max_steps=100, distance_threshold=10):
        """
        🎯 GO TO TARGET - Navigate to the set target destination using RL agent
        
        REAL GAME USAGE:
        ----------------
        # In multi_agent_test.py:
        agent_town_to_anvil = Agent(...)
        agent_town_to_anvil.load_models("town_to_anvil")
        agent_town_to_anvil.set_target_destination("anvil")
        
        # Navigate to target
        success = agent_town_to_anvil.go_target(game_controller)
        if success:
            print("Successfully reached anvil!")
        
        PARAMETERS:
        -----------
        game_controller: Game controller instance for getting position and executing actions
        max_steps (int): Maximum navigation steps before timeout (default: 100)
        distance_threshold (float): Distance to consider target reached (default: 10)
        
        RETURNS:
        --------
        bool: True if target reached successfully, False if failed/timeout
        """
        if not hasattr(self, 'current_target'):
            print("❌ No target set! Use set_target_destination() first.")
            return False
        
        if not hasattr(self, 'loaded_path'):
            print("❌ No model loaded! Use load_models() first.")
            return False
        
        # Check current position
        current_pos = game_controller.get_current_position()
        if not current_pos:
            print("❌ Could not read current position")
            return False
        
        print(f"📍 Current position: {current_pos}")
        
        # Get expected start position for this path
        path_start_positions = {
            "town_to_anvil": (822, 533),
            "anvil_to_armor": (812, 605),
            "armor_to_inn": (768, 592),
            "inn_to_armor": (764, 648),
            "armor_to_anvil": (767, 599)
        }
        
        expected_start = path_start_positions.get(self.loaded_path)
        if expected_start:
            distance_to_start = np.sqrt((current_pos[0] - expected_start[0])**2 + (current_pos[1] - expected_start[1])**2)
            
            if distance_to_start > 50:
                print(f"❌ Too far from expected start position {expected_start}")
                print(f"   Current distance: {distance_to_start:.1f} units")
                print(f"   This agent was trained for path starting at {expected_start}")
                print(f"   Move closer to start position and try again")
                return False
            
            print(f"✅ Position is good! Distance to start: {distance_to_start:.1f} units")
        
        print(f"🎯 Starting navigation to {self.current_destination} at {self.current_target}")
        print(f"📊 Using model: {self.loaded_path}")
        
        steps = 0
        target_x, target_y = self.current_target
        
        # Action mapping for real game control - UPDATED FOR 3 ACTIONS
        action_map = {
            0: 'w',  # Forward
            1: 'a',  # Left  
            2: 'd'   # Right
            # Removed: 3: 's' (Backward) - not used in 3-action system
        }
        
        while steps < max_steps:
            try:
                # Get current position from game
                current_pos = game_controller.get_current_position()
                if current_pos is None:
                    print("❌ Could not get current position from game")
                    return False
                
                current_x, current_y = current_pos
                
                # Calculate distance to target
                distance_to_target = np.sqrt((current_x - target_x)**2 + (current_y - target_y)**2)
                
                # Check if target reached
                if distance_to_target <= distance_threshold:
                    print(f"✅ Target reached! Distance: {distance_to_target:.2f}")
                    return True
                
                # IMPORTANT: State calculation should be done by the navigation system
                # This simplified approach is kept for compatibility but should be avoided
                # Prefer using MultiAgentNavigationSystem.navigate_with_agent() instead
                
                # Create simplified state vector (normalized)
                dx = (target_x - current_x) / 100.0  # Normalize
                dy = (target_y - current_y) / 100.0  # Normalize
                dist_normalized = distance_to_target / 100.0
                
                # Create 12-dimensional state vector similar to training
                # NOTE: This is a simplified version - proper state should include histories
                state = np.array([
                    dx, dy, dist_normalized,
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Velocity and action history placeholders
                    dx, dy, dist_normalized  # Additional features
                ], dtype=np.float32)
                
                # Get action from agent
                action = self.choose_action(state)
                action_key = action_map[action]
                
                # Execute action in game
                game_controller.execute_action(action_key)
                
                print(f"Step {steps+1}: Pos({current_x:.1f},{current_y:.1f}) -> Target({target_x},{target_y}) | Distance: {distance_to_target:.2f} | Action: {action_key}")
                
                steps += 1
                
                # Small delay between actions
                import time
                time.sleep(0.1)
                
            except Exception as e:
                print(f"❌ Error during navigation: {e}")
                return False
        
        print(f"❌ Navigation timeout after {max_steps} steps")
        return False
    