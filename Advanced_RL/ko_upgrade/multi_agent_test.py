# -*- coding: utf-8 -*-
"""
Created on Sat Jul 26 17:48:26 2025

@author: ismt
"""

# -*- coding: utf-8 -*-
"""
🤖 Multi-Agent Path Navigation System for Knight Online
Load and use different trained agents for different paths with go_target() interface
"""

import numpy as np
import time
import os
from game_controller import GameController
from get_window import WindowCapture
from agent_new import Agent
import tensorflow as tf

# Disable GPU if needed
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class MultiAgentNavigationSystem:
    def __init__(self):
        """Initialize the multi-agent navigation system"""
        print("🤖 Initializing Multi-Agent Navigation System...")
        
        # Initialize game interface
        self.controller = GameController("Knight Online Client")
        self.wincap = WindowCapture("Knight Online Client")
        
        # Add get_current_position method to controller
        self.controller.get_current_position = self._get_current_position
        self.controller.execute_action = self._execute_action
        
        # OCR area for coordinates
        self.coord_area = [(44, 105), (163, 122)]
        
        # World parameters - SAME AS RL_TEST
        self.world_size = 1000
        
        # State tracking - CONSISTENT WITH RL_TEST (12D)
        self.last_position = None
        self.current_position = None
        self.current_target = (0, 0)  # Will be set by path
        
        # State history for 12D state vector (consistent with rl_test)
        self.dx_history = []  # Target x distance history
        self.dy_history = []  # Target y distance history
        self.action_history = []  # Action history
        self.velocity_x = 0
        self.velocity_y = 0
        
        # Behavioral tracking
        self.oscillation_count = 0
        self.stuck_count = 0
        
        # Initialize step counter
        self.current_step = 0
        
        # Consecutive rotation tracking for automatic forward insertion
        self.consecutive_rotations = 0
        self.last_executed_action = -1
        
        # Available trained agents and their paths
        self.available_agents = {
            "town_to_anvil": {"start": (822, 533), "end": (817, 603)},
            "anvil_to_armor": {"start": (812, 605), "end": (767, 597)},
            "armor_to_inn": {"start": (768, 592), "end": (764, 648)},
            "inn_to_armor": {"start": (764, 648), "end": (766, 600)},
            "armor_to_anvil": {"start": (767, 599), "end": (818, 605)}
        }
        
        # Store loaded agents nd": (765, 597)},
        self.agents = {}
        
        print(f"🗺️ Available trained agents: {list(self.available_agents.keys())}")
    
    def read_location(self):
        """Read current location using OCR - SAME AS RL_TEST"""
        screenshot = self.wincap.get_screenshot()
        if screenshot is None:
            return None
            
        location_text = self.controller.read_location_with_ocr(screenshot, [self.coord_area])
        
        if location_text and location_text != "❌ Koordinat bulunamadı":
            try:
                parts = location_text.split(',')
                if len(parts) == 2:
                    x = int(parts[0])
                    y = int(parts[1])
                    return (x, y)
            except:
                pass
        
        return None
    
    def _get_current_position(self):
        """Get current position using OCR"""
        return self.read_location()
    
    def _execute_action(self, action_key):
        """Execute action in game - SAME AS RL_TEST"""
        try:
            # Map actions to durations for consistent movement - SAME AS RL_TEST
            action_durations = {
                'w': 1.18 * (3/5),  # ~0.7s for 3 units (SAME AS RL_TEST)
                'a': 0.5,           # 30° rotation (SAME AS RL_TEST)
                'd': 0.75            # 30° rotation (SAME AS RL_TEST)
            }
            
            duration = action_durations.get(action_key, 0.5)
            self.controller.press_(action_key, hold_duration=duration)
            
        except Exception as e:
            print(f"❌ Error executing action {action_key}: {e}")
    
    def calculate_distance(self, pos1, pos2):
        """Calculate distance between two positions - SAME AS RL_TEST"""
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        return np.sqrt(dx * dx + dy * dy)
    
    def initialize_histories(self, target_coords):
        """Initialize histories for 12D state vector (consistent with rl_test)"""
        if self.current_position:
            # Set current target
            self.current_target = target_coords
            
            # Initialize dx/dy histories (maintain 3 elements like rl_test)
            dx = self.current_target[0] - self.current_position[0]
            dy = self.current_target[1] - self.current_position[1]
            
            self.dx_history = [dx, dx, dx]  # 3 elements
            self.dy_history = [dy, dy, dy]  # 3 elements
            
            # Action history - start with -1 (no action) like rl_test
            self.action_history = [-1, -1, -1]
            
            # Velocity tracking
            self.velocity_x = 0
            self.velocity_y = 0
            
            # Behavioral tracking
            self.oscillation_count = 0
            self.stuck_count = 0
            
            print(f"📊 Histories initialized for 12D state:")
            print(f"   Current target: {self.current_target}")
            print(f"   Initial dx: {dx:.1f}")
            print(f"   Initial dy: {dy:.1f}")
            print(f"   Distance to target: {self.calculate_distance(self.current_position, self.current_target):.1f}")
    
    def get_state(self):
        """Generate 12D state vector (EXACTLY consistent with rl_test)"""
        if not self.current_position:
            return np.zeros(12, dtype=np.float32)
        
        # Current relative distances
        current_dx = self.current_target[0] - self.current_position[0]
        current_dy = self.current_target[1] - self.current_position[1]
        
        # Distance to target
        distance = np.sqrt(current_dx**2 + current_dy**2)
        
        # Use EXACT same normalization as rl_test
        max_distance = 200  # This matches rl_test's max(world_width, world_height)
        
        # Debug info with CONSISTENCY CHECK
        current_step = getattr(self, 'current_step', 0)
        if current_step <= 3 or current_step % 10 == 0:
            # print(f"    🔍 MULTI-AGENT Raw values:")
            # print(f"       current_dx: {current_dx:.1f}, current_dy: {current_dy:.1f}")
            print(f"       distance: {distance:.1f}")
            # print(f"       velocity_x: {self.velocity_x:.1f}, velocity_y: {self.velocity_y:.1f}")
            # print(f"       action_history: {self.action_history}")
            # print(f"       dx_history: {self.dx_history}")
            # print(f"       dy_history: {self.dy_history}")
            
        
        # Action history with EXACT same logic as rl_test (updated for 3 actions)
        if len(self.action_history) >= 3:
            act_0 = self.action_history[0] / 2.0 if self.action_history[0] >= 0 else -0.33
            act_1 = self.action_history[1] / 2.0 if self.action_history[1] >= 0 else -0.33  
            act_2 = self.action_history[2] / 2.0 if self.action_history[2] >= 0 else -0.33
        else:
            act_0 = act_1 = act_2 = -0.33  # No action indicator
        
        # dx/dy history with EXACT same logic as rl_test
        if len(self.dx_history) >= 3 and len(self.dy_history) >= 3:
            dx_0 = self.dx_history[0] / max_distance  # Current dx (index 0 = newest)
            dx_1 = self.dx_history[1] / max_distance  # Previous dx (index 1)  
            dx_2 = self.dx_history[2] / max_distance  # Previous-1 dx (index 2)
            dy_0 = self.dy_history[0] / max_distance  # Current dy (index 0 = newest)
            dy_1 = self.dy_history[1] / max_distance  # Previous dy (index 1)
            dy_2 = self.dy_history[2] / max_distance  # Previous-1 dy (index 2)
        else:
            dx_0 = dx_1 = dx_2 = current_dx / max_distance
            dy_0 = dy_1 = dy_2 = current_dy / max_distance
        
        # EXACT SAME ORDER AS RL_TEST
        state = [
            distance / max_distance,         # [0] Normalized distance
            self.velocity_x / 5.0,           # [1] Normalized velocity x
            self.velocity_y / 5.0,           # [2] Normalized velocity y
            act_0,                           # [3] Last action normalized
            act_1,                           # [4] Action t-1 normalized
            act_2,                           # [5] Action t-2 normalized
            dx_0,                            # [6] Current dx
            dx_1,                            # [7] Previous dx (t-1)
            dx_2,                            # [8] Previous-1 dx (t-2)
            dy_0,                            # [9] Current dy
            dy_1,                            # [10] Previous dy (t-1)
            dy_2                             # [11] Previous-1 dy (t-2)
        ]
        
        return np.array(state, dtype=np.float32)
    
    def execute_action_and_update_histories(self, action):
        """Execute action and update histories EXACTLY like rl_test"""
        action_keys = ['w', 'a', 'd']  # Removed 's' (backward)
        action_names = ['Forward', 'Left', 'Right']  # Removed 'Backward'
        
        # Track consecutive rotations and auto-insert forward movement
        if action in [1, 2]:  # Rotation actions (A or D)
            self.consecutive_rotations += 1
            #print(f"   🔄 Consecutive rotations: {self.consecutive_rotations}")
            
            # After 6 consecutive rotations, force a forward movement
            if self.consecutive_rotations >= 6:
                #print(f"   ⚡ 6 consecutive rotations detected! Auto-inserting forward movement after this rotation.")
                # Execute the current rotation first
                self._execute_single_action(action)
                # Then auto-execute forward movement
                #print(f"   ⬆️ Auto-executing forward movement (W)...")
                self._execute_single_action(0)  # Forward action
                self.consecutive_rotations = 0  # Reset counter
                return
        else:
            # Reset consecutive rotation counter on non-rotation actions
            self.consecutive_rotations = 0
        
        # Execute the single action normally
        self._execute_single_action(action)
    
    def _execute_single_action(self, action):
        """Execute a single action with history updates - SAME AS RL_TEST"""
        action_keys = ['w', 'a', 'd']  # Removed 's' (backward)
        action_names = ['Forward', 'Left', 'Right']  # Removed 'Backward'
        
        # Store position before action (like rl_test)
        self.last_position = self.current_position
        
        # Update action history EXACTLY like rl_test (after storing position)
        self.action_history = [action] + self.action_history[:2]
        
        # Update dx/dy history EXACTLY like rl_test (after storing position)
        current_dx = self.current_target[0] - self.current_position[0]
        current_dy = self.current_target[1] - self.current_position[1]
        self.dx_history = [current_dx] + self.dx_history[:2]
        self.dy_history = [current_dy] + self.dy_history[:2]
        
        #print(f"   Pre-action dx/dy: {current_dx:.1f}, {current_dy:.1f}")
        #print(f"   Updated dx_history: {self.dx_history}")
        #print(f"   Updated dy_history: {self.dy_history}")
        
        # CRITICAL: Match rl_test movement distances as closely as possibleh
        if action == 0:  # W (Forward) - target 3 units
            hold_duration = 1.18 * (3/5)  # ~0.7s for 3 units
        elif action == 1:
            hold_duration = 0.5
        elif action == 2:  # A/D (Rotation) - 30° equivalent
            hold_duration = 0.75  # 30° rotation
        else:
            hold_duration = 1.0  # Default
        
        print(f"  Action: {action_names[action]} ({action_keys[action].upper()}) - Duration: {hold_duration}s")
        
        # Press key
        self.controller.press_(action_keys[action], hold_duration=hold_duration)
        
        # Update last executed action
        self.last_executed_action = action
    
    def target_reached(self, target_coords, distance_threshold=3):
        """Check if target is reached - SAME AS RL_TEST"""
        if not self.current_position:
            return False
            
        distance_to_target = self.calculate_distance(
            self.current_position,
            target_coords
        )
        
        return distance_to_target < distance_threshold
    
    def load_agent(self, path_name):
        """Load a specific trained agent for a path"""
        if path_name not in self.available_agents:
            print(f"❌ Unknown path: {path_name}")
            print(f"   Available paths: {list(self.available_agents.keys())}")
            return None
        
        if path_name in self.agents:
            print(f"✅ Agent {path_name} already loaded")
            return self.agents[path_name]
        
        try:
            print(f"🔄 Loading agent for path: {path_name}")
            
            # Create agent with same parameters as training
            agent = Agent(
                gamma=0.99,
                epsilon=0.01,  # Low epsilon for testing
                lr=0.0005,
                n_actions=3,   # W, A, D (removed S/backward)
                input_dims=(12,),  # 12D state vector
                mem_size=50000,
                batch_size=64,
                eps_min=0.01,
                eps_dec=0,  # No decay during testing
                replace=1000,
                algo='DQN',
                env_name='NavigationEnv',
                chkpt_dir='models_real/'
            )
            
            # Load the specific model for this path
            agent.load_models(path_name=path_name)
            
            # Store the loaded agent
            path_info = self.available_agents[path_name]
            self.agents[path_name] = agent
            
            print(f"✅ Agent {path_name} loaded successfully")
            print(f"   Start: {path_info['start']}")
            print(f"   End: {path_info['end']}")
            
            return agent
            
        except Exception as e:
            print(f"❌ Failed to load agent {path_name}: {e}")
            return None
    
    def load_all_agents(self):
        """Load all available trained agents"""
        print("🔄 Loading all trained agents...")
        
        success_count = 0
        for path_name in self.available_agents.keys():
            agent = self.load_agent(path_name)
            if agent:
                success_count += 1
        
        print(f"✅ Loaded {success_count}/{len(self.available_agents)} agents successfully")
        return success_count == len(self.available_agents)
    
    def get_agent(self, path_name):
        """Get a loaded agent by path name"""
        if path_name not in self.agents:
            print(f"⚠️ Agent {path_name} not loaded. Loading now...")
            return self.load_agent(path_name)
        
        return self.agents[path_name]
    
    def navigate_with_agent(self, path_name, max_steps=500, distance_threshold=4):
        """Navigate using a specific agent - SAME PATTERN AS RL_TEST"""
        print(f"\n🎯 Starting navigation with agent: {path_name}")
        
        # Get or load the agent
        agent = self.get_agent(path_name)
        if not agent:
            print(f"❌ Failed to get agent for {path_name}")
            return False
        
        # Initialize position
        print("📍 Reading initial position...")
        for i in range(5):
            pos = self.read_location()
            if pos:
                self.current_position = pos
                self.last_position = pos
                print(f"✅ Initial position: {pos}")
                break
            time.sleep(0.5)
        
        if not self.current_position:
            print("❌ Failed to read initial position!")
            return False
        
        path_info = self.available_agents[path_name]
        start_pos = path_info["start"]
        target_coords = path_info["end"]
        
        # Check if we're near the expected start position
        start_distance = self.calculate_distance(self.current_position, start_pos)
        
        if start_distance > 50:
            print(f"⚠️ WARNING: Current position {self.current_position} is {start_distance:.1f} units from expected start {start_pos}")
            print(f"   This agent was trained for the path {start_pos} → {target_coords}")
            print(f"   Navigation may not work optimally from current position")
            return False
        
        # Initialize histories PROPERLY - SAME AS RL_TEST
        self.initialize_histories(target_coords)
        
        print(f"📏 Initial distance to target:")
        print(f"   {self.calculate_distance(self.current_position, target_coords):.1f} units")
        print("\n🎮 Starting navigation to target...")
        print("-" * 40)
        
        # Main loop - SAME AS RL_TEST
        step = 0
        time.sleep(3)
        start_time = time.time()
        
        while step < max_steps:
            step += 1
            
            # Get current state (BEFORE action, like rl_test)
            state = self.get_state()
            
            # Debug state (first few steps and every 10 steps)
            if step <= 3 or step % 10 == 0:
                print(f"    🧠 MULTI-AGENT STATE VECTOR (12D):")
                print(f"    {state}")
                
            # Get action from agent
            action = agent.choose_action(state)
            
            # Store step for debugging
            self.current_step = step
            
            # Execute action - SAME AS RL_TEST
            print(f"\nStep {step}:")
            self.execute_action_and_update_histories(action)
            
            # Wait for movement
            time.sleep(0.05)
            
            # Read new position
            new_pos = self.read_location()
            
            if new_pos:
                self.current_position = new_pos
                
                # Update velocity after movement (EXACTLY like rl_test)
                if self.last_position:
                    self.velocity_x = self.current_position[0] - self.last_position[0]
                    self.velocity_y = self.current_position[1] - self.last_position[1]
                    
                    print(f"   Post-action position: {self.current_position}")
                    print(f"   Post-action velocity: ({self.velocity_x:.1f}, {self.velocity_y:.1f})")
                    print(f"   Post-action dx/dy: {self.current_target[0] - self.current_position[0]:.1f}, {self.current_target[1] - self.current_position[1]:.1f}")
                
                # Calculate distance to target
                distance_to_target = self.calculate_distance(self.current_position, target_coords)
                
                print(f"  Position: {self.current_position}")
                print(f"  Distance to target: {distance_to_target:.1f}")
                
                # Check if target reached (SUCCESS!)
                if self.target_reached(target_coords, distance_threshold):
                    elapsed = time.time() - start_time
                    print("\n" + "="*60)
                    print("🎯 SUCCESS! TARGET REACHED!")
                    print(f"✅ Agent path: {path_name}")
                    print(f"✅ Start: {start_pos}")
                    print(f"✅ Target: {target_coords}")
                    print(f"✅ Steps: {step}")
                    print(f"⏱️ Time: {elapsed:.1f} seconds")
                    print(f"🎯 Final distance: {distance_to_target:.1f}")
                    print("="*60)
                    return True
            else:
                print("  ⚠️ OCR failed, using last position")
                # OCR failed - update velocity to 0
                self.velocity_x = 0
                self.velocity_y = 0
                print(f"   OCR failed - velocity set to 0")
            
            # Print progress every 20 steps
            if step % 20 == 0:
                distance_to_target = self.calculate_distance(self.current_position, target_coords)
                
                print(f"\n📊 Progress Report:")
                print(f"   Steps: {step}/{max_steps}")
                print(f"   Path: {path_name}")
                print(f"   Start: {start_pos}")
                print(f"   Target: {target_coords}")
                print(f"   Distance to target: {distance_to_target:.1f}")
                print(f"   Time: {time.time() - start_time:.1f}s")
                print("-" * 40)
        
        # Time out
        distance_to_target = self.calculate_distance(self.current_position, target_coords)
        print("\n" + "="*60)
        print("⏰ TIME OUT! Maximum steps reached.")
        print(f"❌ Path: {path_name}")
        print(f"❌ Start: {start_pos}")
        print(f"❌ Target: {target_coords}")
        print(f"❌ Final distance: {distance_to_target:.1f}")
        print(f"❌ Target radius: {distance_threshold}")
        print("="*60)
        return False
    
    def list_available_agents(self):
        """List all available trained agents"""
        print("\n🤖 Available Trained Agents:")
        print("=" * 50)
        
        for i, (path_name, path_info) in enumerate(self.available_agents.items(), 1):
            loaded_status = "✅ Loaded" if path_name in self.agents else "⭕ Not loaded"
            print(f"{i}. {path_name}")
            print(f"   Path: {path_info['start']} → {path_info['end']}")
            print(f"   Status: {loaded_status}")
            print()
