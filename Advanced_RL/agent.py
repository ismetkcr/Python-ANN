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

    def load_models(self):
        """Load models with compatibility fixes for newer TensorFlow versions"""
        try:
            # Try loading with custom objects for compatibility
            import tensorflow as tf
            from tensorflow import keras
            
            print("Loading models...")
            
            # Custom objects for compatibility
            custom_objects = {}
            
            # Load models with error handling
            # self.Q = keras.models.load_model(trained_model_path, compile=False)
            # self.Q.compile(optimizer=keras.optimizers.Adam(learning_rate=self.lr))
            try:
                self.q_eval = keras.models.load_model(
                    self.fname + 'q_eval.h5',
                    custom_objects=custom_objects,
                    compile=False  # Don't compile initially
                )
                print("✅ q_eval model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading q_eval: {e}")
                raise
                
            try:
                self.q_next = keras.models.load_model(
                    self.fname + 'q_next.h5',
                    custom_objects=custom_objects,
                    compile=False  # Don't compile initially
                )
                print("✅ q_next model loaded successfully")
            except Exception as e:
                print(f"❌ Error loading q_next: {e}")
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
            
            print("✅ Models recompiled with current optimizer settings")
            print(f"✅ Learning rate set to: {self.lr}")
            
        except Exception as e:
            print(f"❌ Failed to load models: {e}")
            print("💡 TROUBLESHOOTING TIPS:")
            print("   1. Check if model files exist in 'models/' directory")
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
        
        USAGE IN REAL GAME:
        -------------------
        # In RL_in_real_game/rl_test.py:
        path_coords = agent.get_path_coordinates("town_to_anvil")
        start_pos = path_coords[0]    # Starting position
        end_pos = path_coords[-1]     # Target position
        waypoints = path_coords[1:-1] # Intermediate waypoints
        
        PARAMETERS:
        -----------
        path_name (str): Name of the path (use get_available_paths() to see options)
        
        RETURNS:
        --------
        list: List of (x, y) coordinate tuples for the path
        Example: [(822, 533), (814, 555), (814, 570), (814, 590), (814, 604)]
        """
        path_definitions = {
            "town_to_anvil": [
                (822, 533),  # Start: Town
                (814, 555),  # Enter right corridor area  
                (814, 570),  # Middle of right corridor
                (814, 590),  # Exit corridor area
                (814, 604)   # End: Anvil
            ],
            "anvil_to_armor": [
                (812, 605),  # Start: Anvil area
                (801, 603),  # Waypoint 1
                (786, 600),  # Waypoint 2
                (773, 597),  # Waypoint 3
                (767, 599)   # End: Armor
            ],
            "armor_to_inn": [
                (766, 600),  # Start: Armor
                (766, 614),  # Waypoint 1
                (765, 634),  # Waypoint 2
                (764, 648)   # End: Inn
            ],
            "inn_to_armor": [
                (764, 648),  # Start: Inn
                (765, 634),  # Waypoint 1 (reverse of armor_to_inn)
                (766, 614),  # Waypoint 2
                (766, 600)   # End: Armor
            ],
            "armor_to_anvil": [
                (767, 599),  # Start: Armor
                (773, 597),  # Waypoint 1 (reverse of anvil_to_armor)
                (786, 600),  # Waypoint 2
                (801, 603),  # Waypoint 3
                (812, 605)   # End: Anvil area
            ]
        }
        
        if path_name not in path_definitions:
            available = list(path_definitions.keys())
            raise ValueError(f"Unknown path '{path_name}'. Available: {available}")
        
        return path_definitions[path_name]
    
    def get_destination_coordinates(self, destination):
        """
        🎯 GET DESTINATION COORDINATES - Returns target coordinates for real game
        
        REAL GAME USAGE:
        ----------------
        # In RL_in_real_game/rl_test.py:
        target_coords = agent.get_destination_coordinates("anvil")
        # Returns: (814, 604) - anvil coordinates
        
        AVAILABLE DESTINATIONS:
        -----------------------
        - "anvil"  : Blacksmith/Anvil (814, 604)  
        - "armor"  : Armor Shop (766, 600)
        - "inn"    : Inn (764, 648)
        
        NOTE: Town (822, 533) is the START POINT, not a destination
        
        PARAMETERS:
        -----------
        destination (str): Where to go ("anvil", "armor", "inn")
        
        RETURNS:
        --------
        tuple: (x, y) coordinates of the destination
        """
        destinations = {
            "anvil": (814, 604),  # Blacksmith/Anvil
            "armor": (766, 600),  # Armor Shop  
            "inn": (764, 648)     # Inn
        }
        
        if destination not in destinations:
            available = list(destinations.keys())
            raise ValueError(f"Unknown destination '{destination}'. Available: {available}")
        
        return destinations[destination]
    
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
    
    