import numpy as np
import pygame
import sys

class UltraSimpleEnvironment:
    """
    REAL GAME ENVIRONMENT:
    - Agent spawns near initial real game location
    - Target is set to given real game coordinates
    - Uses normalized coordinates to work with real game data
    - Prevents agent from going too far from spawn area
    """
    
    def __init__(self, render_mode=True, real_game_path=None, path_name=None, destination=None):
        # UNIVERSAL PATH SYSTEM: Agent can handle any specified path
        
        # Define all available paths for universal training/usage
        self.all_available_paths = {
            "town_to_anvil": [
                (822, 533),  # Start: Town
                (814, 555),  # Enter right corridor area  
                (814, 570),  # Middle of right corridor
                (814, 590),  # Exit corridor area
                (817, 604)   # End: Anvil
            ],
            "anvil_to_armor": [
                (825, 605),
                (812, 605),  # Start: Anvil area
                (801, 603),  # Waypoint 1
                (786, 600),  # Waypoint 2
                (773, 597),  # Waypoint 3
                (767, 599)   # End: Armor
            ],
            "armor_to_inn": [
                (768, 592),  # Start: Armor
                (768, 614),  # Waypoint 1
                (768, 634),  # Waypoint 2
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
        
        # Set the current path - Support both path_name and destination modes
        if destination:
            # DESTINATION MODE: Train agent to go to a specific destination
            self.current_path_name = f"to_{destination}"
            self.destination_name = destination
            self.real_game_path = self._get_path_to_destination(destination)
            print(f"🎯 Environment initialized for destination: {destination}")
            print(f"   Route: {self.real_game_path[0]} -> {self.real_game_path[-1]}")
        elif real_game_path:
            self.real_game_path = real_game_path
            self.current_path_name = path_name if path_name else "custom"
            print(f"🎯 Environment initialized for custom path: {self.current_path_name}")
        elif path_name and path_name in self.all_available_paths:
            self.real_game_path = self.all_available_paths[path_name]
            self.current_path_name = path_name
            print(f"🎯 Environment initialized for path: {self.current_path_name}")
            print(f"   Route: {self.real_game_path[0]} -> {self.real_game_path[-1]}")
        else:
            # Default to town_to_anvil for single-path training
            self.real_game_path = self.all_available_paths["town_to_anvil"]
            self.current_path_name = "town_to_anvil"
            print(f"🎯 Environment initialized for default path: {self.current_path_name}")
            print(f"   Route: {self.real_game_path[0]} -> {self.real_game_path[-1]}")
        
        # Extract initial and target locations from path
        self.initial_location = self.real_game_path[0]
        self.target_location = self.real_game_path[-1]
        
        # Define boundaries around initial spawn (prevent going too far)
        self.boundary_range = 100  # Allow 100 units deviation from spawn
        
        # Calculate world bounds based on spawn location
        spawn_x, spawn_y = self.initial_location
        self.world_x_min = spawn_x - self.boundary_range
        self.world_x_max = spawn_x + self.boundary_range
        self.world_y_min = spawn_y - self.boundary_range
        self.world_y_max = spawn_y + self.boundary_range
        
        # World size for normalization
        self.world_width = self.world_x_max - self.world_x_min
        self.world_height = self.world_y_max - self.world_y_min
        
        # Movement parameters
        self.forward_step_size = 3  # W: 3 units/step
        self.rotation_angle = 30    # A/D: 30°/step
        
        # Episode length
        self.max_steps = 250
        
        # Action space: 0=W, 1=A, 2=D (removed S/backward)
        self.n_actions = 3
        
        # Target (set from real game coordinates)
        self.target_x = self.target_location[0]
        self.target_y = self.target_location[1]
        self.target_radius = 5  # Precise target for real game difficulty
        
        # Agent visual properties
        self.agent_display_radius = 8  # Agent size in pixels (was 10, now smaller)
        
        # Agent (will spawn near initial location)
        self.agent_x = 0
        self.agent_y = 0
        self.agent_angle = 0  # For movement direction
        
        # Episode tracking
        self.current_step = 0
        self.last_action = -1
        
        # State history for relative distances (last 4 values)
        self.dx_history = []
        self.dy_history = []
        
        # Pygame rendering
        self.render_mode = render_mode
        if self.render_mode:
            pygame.init()
            self.window_size = 600
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Ultra Simple Navigation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
        
        self.reset()
    
    def reset(self):
        """Reset environment with agent spawning near initial real game location"""
        # Use the path that was set during initialization - NO RANDOM PATH CHANGES
        # This ensures single-path training consistency
        
        # Spawn agent near initial location with small randomization
        spawn_x, spawn_y = self.initial_location
        
        # CAREFUL SPAWN CALCULATION: Respect X=762 boundary from the start
        # Calculate safe spawn range that respects boundary
        safe_spawn_x_min = max(762, spawn_x - 8)  # Don't go below X=762
        safe_spawn_x_max = min(self.world_x_max, spawn_x + 8)  # Don't exceed world bounds
        
        # Generate spawn position within safe boundaries
        if safe_spawn_x_max > safe_spawn_x_min:
            self.agent_x = int(np.random.randint(safe_spawn_x_min, safe_spawn_x_max + 1))
        else:
            self.agent_x = safe_spawn_x_min  # Default to minimum safe position
            
        self.agent_y = int(spawn_y + np.random.randint(-8, 9))
        
        # Ensure agent stays within world bounds (integer coordinates)
        self.agent_x = int(np.clip(self.agent_x, 762, self.world_x_max))  # Use 762 as minimum instead of world_x_min
        self.agent_y = int(np.clip(self.agent_y, self.world_y_min, self.world_y_max))
        
        # Ensure agent spawns within critical distance of path
        agent_pos = np.array([self.agent_x, self.agent_y])
        min_distance_to_path = float('inf')
        for i in range(len(self.real_game_path) - 1):
            point1 = np.array(self.real_game_path[i])
            point2 = np.array(self.real_game_path[i + 1])
            distance = self._point_to_line_distance(agent_pos, point1, point2)
            if distance < min_distance_to_path:
                min_distance_to_path = distance
        
        # If agent spawns too far from path, move it closer
        if min_distance_to_path > 3.0:  # Stricter safety margin
            # Find closest point on path and move agent there
            closest_point = self.real_game_path[0]  # Default to first point
            for point in self.real_game_path:
                point_distance = np.sqrt((self.agent_x - point[0])**2 + (self.agent_y - point[1])**2)
                if point_distance < np.sqrt((self.agent_x - closest_point[0])**2 + (self.agent_y - closest_point[1])**2):
                    closest_point = point
            
            # Move agent to closest path point with small randomization (integer coordinates)
            self.agent_x = int(closest_point[0] + np.random.randint(-3, 4))
            self.agent_y = int(closest_point[1] + np.random.randint(-3, 4))
        
        # Target remains fixed at the end of the path
        self.target_x = self.target_location[0]
        self.target_y = self.target_location[1]
        
        # Random starting angle
        self.agent_angle = np.random.randint(0, 360)  # Random facing direction
        
        self.current_step = 0
        self.last_action = -1
        
        # Initialize previous position for velocity calculation
        self.prev_x = self.agent_x
        self.prev_y = self.agent_y
        
        # Initialize action history
        self.action_history = [-1, -1, -1]
        
        # Position history for stuck detection
        self.position_history = [(self.agent_x, self.agent_y)]
        
        # Reset collision flag
        self.collision_occurred = False
        
        # Reset history with current relative distances
        dx = self.target_x - self.agent_x
        dy = self.target_y - self.agent_y
        self.dx_history = [dx, dx, dx]  # Initialize with current dx (3 values)
        self.dy_history = [dy, dy, dy]  # Initialize with current dy (3 values)
        
        # Reset path progress tracking
        self.last_segment_index = 0
        
        # Reset path violation flag
        self.path_violation = False
        
        # PROGRESS TRACKING
        self.last_distance_to_target = self._distance_to_target()
        self.best_distance_to_target = self._distance_to_target()  # Track best distance achieved
        self.stuck_counter = 0  # Count steps without progress
        
        # TARGET VISIT TRACKING - prevent reward abuse
        self.target_visited = False
        self.closest_distance_achieved = float('inf')
        
        # ANTI-BACKTRACKING SYSTEM
        self.progress_history = []  # Track recent progress toward target
        self.forward_momentum = 0   # Track forward movement momentum
        
        # SPAWN AREA BEHAVIOR TRACKING
        self.steps_at_spawn = 0     # Count steps spent near spawn area
        
        # ENHANCED STUCK DETECTION
        self.position_buffer = []  # Track recent positions for stuck detection
        self.consecutive_collisions = 0  # Count consecutive collision attempts
        self.last_successful_move_step = 0  # Track last time agent successfully moved
        self.blocked_direction_penalty = 0.0  # Accumulating penalty for trying blocked directions
        
        # ANTI-CIRCLING SYSTEM
        self.position_history = []  # Track recent positions to detect circling
        self.last_angle = self.agent_angle  # Track angle changes
        self.oscillation_penalty_accumulator = 0.0  # Track oscillation penalties
        
        # FULL ROTATION TRACKING SYSTEM
        self.total_rotation_degrees = 0.0  # Track total rotation since reset
        self.full_rotations_completed = 0  # Count complete 360° rotations
        
        # Define obstacles creating corridors
        self.obstacles = [
            # Obstacle 1 (center) - Original obstacle creating main 2 corridors
            {
                'start': (815, 556),
                'end': (815, 581),
                'thickness': 3
            },
            # Obstacle 2 - Right side boundary obstacle  
            {
                'start': (823, 555),
                'end': (823, 581),
                'thickness': 1
            },
            # Obstacle 3 - Left side boundary obstacle
            {
                'start': (807, 555),
                'end': (807, 582),
                'thickness': 2
            },
            # Obstacle 4 - Inn to Armor corridor obstacle
            {
                'start': (764, 605),
                'end': (764, 642),
                'thickness': 1
            },
            # Obstacle 5 - Additional corridor obstacle
            {
                'start': (770, 601),
                'end': (770, 657),
                'thickness': 1
            },
            # Obstacle 6 - Circle obstacle 1 (new: center 816,606, diameter 2)
            {
                'type': 'circle',
                'center': (820, 592),
                'radius': 3  # diameter 2 = radius 1
            },
            # Obstacle 7 - Circle obstacle 2 (new: center 815,617, diameter 7)
            {
                'type': 'circle',
                'center': (815, 617),
                'radius': 3.5  # diameter 7 = radius 3.5
            },
            
            # Obstacle 7 - Circle obstacle 2 (new: center 815,617, diameter 7)
            {
                'type': 'circle',
                'center': (804, 608),
                'radius': 2 # diameter 7 = radius 3.5
            },
            
            
            # Obstacle 8 - Circle obstacle 3 (center 772,593, diameter 2)
            {
                'type': 'circle',
                'center': (770, 593),
                'radius': 2  # diameter 2 = radius 1
            },
            # Obstacle 9 - Circle obstacle 4 (center 772,601, diameter 2)
            {
                'type': 'circle',
                'center': (780, 601),
                'radius': 2  # diameter 2 = radius 1
            },
            # Obstacle 10 - X=762 boundary wall (spans entire Y dimension)
            {
                'type': 'boundary_wall',
                'x_position': 762,
                'y_start': self.world_y_min,
                'y_end': self.world_y_max,
                'thickness': 1
            },
            
            
            # Obstacle 11 - Circle obstacle 3 (center 772,593, diameter 2)
            {
                'type': 'circle',
                'center': (763, 644),
                'radius': 2  # diameter 2 = radius 1
            },
            
            # Obstacle 12 - Circle obstacle 3 (center 772,593, diameter 2)
            {
                'type': 'circle',
                'center': (815, 581),
                'radius': 4  # diameter 2 = radius 1
            },
            
        ]
        
        # Legacy properties for backward compatibility
        self.obstacle_start = self.obstacles[0]['start']
        self.obstacle_end = self.obstacles[0]['end']
        self.obstacle_thickness = self.obstacles[0]['thickness']
        
        # Define corridor boundaries for violation detection
        self.left_corridor_x_min = 808   # Left boundary of left corridor
        self.left_corridor_x_max = 816   # Right boundary of left corridor (center obstacle)
        self.right_corridor_x_min = 816  # Left boundary of right corridor (center obstacle)
        self.right_corridor_x_max = 823  # Right boundary of right corridor
        self.corridor_y_min = 555        # Top of corridor area
        self.corridor_y_max = 582        # Bottom of corridor area
        
        # print(f"🚧 Created 8 obstacles forming corridors:")
        # print(f"   Left corridor: X {self.left_corridor_x_min}-{self.left_corridor_x_max}, Y {self.corridor_y_min}-{self.corridor_y_max}")
        # print(f"   Right corridor: X {self.right_corridor_x_min}-{self.right_corridor_x_max}, Y {self.corridor_y_min}-{self.corridor_y_max}")
        # print(f"   Agent must stay within corridor boundaries!")
        
        return self._get_state()
    
    def set_path(self, path_name):
        """Set a new path for navigation - useful for real game deployment"""
        if path_name not in self.all_available_paths:
            raise ValueError(f"Unknown path '{path_name}'. Available paths: {list(self.all_available_paths.keys())}")
        
        self.current_path_name = path_name
        self.real_game_path = self.all_available_paths[path_name]
        
        # Update locations
        self.initial_location = self.real_game_path[0]
        self.target_location = self.real_game_path[-1]
        self.target_x = self.target_location[0]
        self.target_y = self.target_location[1]
        
        print(f"🔄 Path changed to: {self.current_path_name}")
        print(f"   New route: {self.real_game_path[0]} -> {self.real_game_path[-1]}")
        
        # Reset episode with new path
        return self.reset()
    
    def get_available_paths(self):
        """Get list of all available paths"""
        return list(self.all_available_paths.keys())
    
    def _get_path_to_destination(self, destination):
        """
        Get path to a specific destination - automatically selects appropriate path
        
        DESTINATION TRAINING:
        --------------------
        destination="anvil" -> Uses town_to_anvil path (spawn near town, go to anvil)
        destination="armor" -> Uses anvil_to_armor path (spawn near anvil, go to armor)
        destination="inn"   -> Uses armor_to_inn path (spawn near armor, go to inn)
        """
        destination_paths = {
            "anvil": "town_to_anvil",      # Train agent to go to anvil from town
            "armor": "anvil_to_armor",     # Train agent to go to armor from anvil  
            "inn": "armor_to_inn"          # Train agent to go to inn from armor
        }
        
        if destination not in destination_paths:
            available = list(destination_paths.keys())
            raise ValueError(f"Unknown destination '{destination}'. Available: {available}")
        
        path_name = destination_paths[destination]
        return self.all_available_paths[path_name]
    
    def _get_state(self):
        """Real-game compatible state: normalized for real game coordinates"""
        # Current relative distances
        current_dx = self.target_x - self.agent_x
        current_dy = self.target_y - self.agent_y
        
        # Distance to target
        distance = np.sqrt(current_dx**2 + current_dy**2)
        
        # Velocity components (change from last position)
        if hasattr(self, 'prev_x') and hasattr(self, 'prev_y'):
            velocity_x = self.agent_x - self.prev_x
            velocity_y = self.agent_y - self.prev_y
        else:
            velocity_x = 0
            velocity_y = 0
        
        # Create normalized state vector (12D) - BACK TO ORIGINAL WORKING VERSION
        # Use action history if available, otherwise use -1 (no action)
        if hasattr(self, 'action_history') and len(self.action_history) >= 3:
            act_0 = self.action_history[0] / 3.0 if self.action_history[0] >= 0 else -0.33
            act_1 = self.action_history[1] / 3.0 if self.action_history[1] >= 0 else -0.33  
            act_2 = self.action_history[2] / 3.0 if self.action_history[2] >= 0 else -0.33
        else:
            act_0 = act_1 = act_2 = -0.33  # No action indicator
        
        # Use dx/dy history with dynamic normalization based on world bounds
        max_distance = max(self.world_width, self.world_height)
        if hasattr(self, 'dx_history') and hasattr(self, 'dy_history') and len(self.dx_history) >= 3:
            dx_0 = self.dx_history[0] / max_distance  # Current dx
            dx_1 = self.dx_history[1] / max_distance  # Previous dx  
            dx_2 = self.dx_history[2] / max_distance  # Previous-1 dx
            dy_0 = self.dy_history[0] / max_distance  # Current dy
            dy_1 = self.dy_history[1] / max_distance  # Previous dy
            dy_2 = self.dy_history[2] / max_distance  # Previous-1 dy
        else:
            dx_0 = dx_1 = dx_2 = current_dx / max_distance
            dy_0 = dy_1 = dy_2 = current_dy / max_distance
        
        # SIMPLE 12D STATE - ORIGINAL WORKING VERSION
        state = [
            distance / max_distance,         # Normalized distance
            velocity_x / 5.0,                # Normalized velocity x
            velocity_y / 5.0,                # Normalized velocity y
            act_0,                           # Last action normalized
            act_1,                           # Action t-1 normalized
            act_2,                           # Action t-2 normalized
            dx_0,                            # Current dx
            dx_1,                            # Previous dx (t-1)
            dx_2,                            # Previous-1 dx (t-2)
            dy_0,                            # Current dy
            dy_1,                            # Previous dy (t-1)
            dy_2                             # Previous-1 dy (t-2)
        ]
        
        return np.array(state, dtype=np.float32)
    
    def step(self, action):
        """Take action"""
        # Store previous position for velocity calculation
        self.prev_x, self.prev_y = self.agent_x, self.agent_y
        prev_x, prev_y = self.agent_x, self.agent_y
        
        # Store the action that will be executed  
        self.last_action = action
        
        # Update action history
        if hasattr(self, 'action_history'):
            self.action_history = [action] + self.action_history[:2]
        
        # Update dx/dy history BEFORE movement (this is the key fix)
        current_dx = self.target_x - self.agent_x
        current_dy = self.target_y - self.agent_y
        if hasattr(self, 'dx_history') and hasattr(self, 'dy_history'):
            self.dx_history = [current_dx] + self.dx_history[:2]
            self.dy_history = [current_dy] + self.dy_history[:2]
        
        # Execute action with collision detection
        collision_occurred = False
        successful_move = False
        old_x, old_y = self.agent_x, self.agent_y  # Store original position
        
        if action == 0:  # W - Forward
            angle_rad = np.radians(self.agent_angle)
            new_x = self.agent_x + self.forward_step_size * np.cos(angle_rad)
            new_y = self.agent_y + self.forward_step_size * np.sin(angle_rad)
            
            # Check for collisions BEFORE moving
            can_move = True
            
            # Check world boundary collision
            if new_x < self.world_x_min or new_x > self.world_x_max or new_y < self.world_y_min or new_y > self.world_y_max:
                can_move = False
                collision_occurred = True
            
            # STRICT X=762 BOUNDARY ENFORCEMENT - Agent cannot cross to x < 762
            if new_x < 762:
                can_move = False
                collision_occurred = True
                #print(f"🚫 BLOCKED: Agent tried to cross x=762 boundary (attempted x={new_x:.1f})")
            
            # Check obstacle collision
            if can_move and self._check_all_obstacles_collision(int(new_x), int(new_y)):
                can_move = False
                collision_occurred = True
            
            # Only move if no collision
            if can_move:
                self.agent_x = int(new_x)
                self.agent_y = int(new_y)
                successful_move = True
                self.last_successful_move_step = self.current_step
            
            # Update position history for stuck detection
            if hasattr(self, 'position_history'):
                self.position_history.append((self.agent_x, self.agent_y))
                # Keep only last 8 positions
                if len(self.position_history) > 8:
                    self.position_history = self.position_history[-8:]
            
        elif action == 1:  # A - Turn left
            self.agent_angle = (self.agent_angle + self.rotation_angle) % 360
            # Track total rotation for full rotation penalty
            self.total_rotation_degrees += self.rotation_angle
            
        elif action == 2:  # D - Turn right
            self.agent_angle = (self.agent_angle - self.rotation_angle) % 360
            # Track total rotation for full rotation penalty
            self.total_rotation_degrees += self.rotation_angle
        
        # Store collision info for reward calculation
        self.collision_occurred = collision_occurred
        
        # ENHANCED STUCK DETECTION TRACKING
        # Update position buffer (keep last 5 positions)
        self.position_buffer.append((self.agent_x, self.agent_y))
        if len(self.position_buffer) > 5:
            self.position_buffer.pop(0)
            
        # Track consecutive collisions
        if collision_occurred:
            self.consecutive_collisions += 1
        else:
            self.consecutive_collisions = 0
            
        # Check if agent is stuck (same position for several steps)
        if len(self.position_buffer) >= 4:
            recent_positions = set(self.position_buffer[-4:])  # Last 4 positions
            if len(recent_positions) == 1:  # All same position = stuck
                self.stuck_counter = self.current_step - self.last_successful_move_step
            else:
                self.stuck_counter = 0
        
        self.current_step += 1
        
        # PROGRESS TRACKING: Check if agent is making progress toward target
        current_distance = self._distance_to_target()
        if current_distance >= self.last_distance_to_target - 0.5:  # No significant progress
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0  # Reset if making progress
        self.last_distance_to_target = current_distance
        
        # Update best distance achieved
        if current_distance < self.best_distance_to_target:
            self.best_distance_to_target = current_distance
        
        # UPDATE POSITION HISTORY for circling detection
        self.position_history.append((self.agent_x, self.agent_y))
        if len(self.position_history) > 8:  # Keep last 8 positions
            self.position_history.pop(0)
        
        # Calculate reward
        reward = self._calculate_reward(prev_x, prev_y)
        
        # Check if done
        done = self._is_done()
        
        # Info
        info = {
            'target_reached': self._target_reached(),
            'distance_to_target': self._distance_to_target(),
            'steps': self.current_step,
            'path_violation': hasattr(self, 'path_violation') and self.path_violation,
            'current_path_name': self.current_path_name
        }
        
        return self._get_state(), reward, done, info
    
    def _calculate_reward(self, prev_x, prev_y):
        """Enhanced reward system with target approach fixes"""
        current_dist = self._distance_to_target()
        
        # TARGET REACHED - Episode success
        if self._target_reached():
            return 1000.0
        
        # CLOSE TO TARGET - Encourage careful approach (NEW FIX)
        if current_dist <= 10.0:  # Very close to target
            # Massive reward for getting closer when near target
            prev_dist = np.sqrt((prev_x - self.target_x)**2 + (prev_y - self.target_y)**2)
            distance_change = prev_dist - current_dist
            
            if distance_change > 0:  # Moving closer
                return 200.0 + distance_change * 100.0  # Huge approach reward
            else:  # Moving away from target when close
                return -100.0  # Strong penalty for moving away when close
        
        # Check if agent is too far from path (essential for proper navigation)
        if len(self.real_game_path) >= 2:
            min_distance_to_path = float('inf')
            agent_pos = np.array([self.agent_x, self.agent_y])
            
            for i in range(len(self.real_game_path) - 1):
                point1 = np.array(self.real_game_path[i])
                point2 = np.array(self.real_game_path[i + 1])
                distance = self._point_to_line_distance(agent_pos, point1, point2)
                if distance < min_distance_to_path:
                    min_distance_to_path = distance
            
            # SAFE ZONE SYSTEM: Don't end game in safe zones, but still give penalties
            spawn_distance = np.sqrt((self.agent_x - self.initial_location[0])**2 + (self.agent_y - self.initial_location[1])**2)
            target_distance = np.sqrt((self.agent_x - self.target_location[0])**2 + (self.agent_y - self.target_location[1])**2)
            
            # Safe zone radii
            spawn_safe_radius = 20.0
            target_safe_radius = 20.0
            
            # Check if agent is in any safe zone
            in_spawn_safe_zone = spawn_distance <= spawn_safe_radius
            in_target_safe_zone = target_distance <= target_safe_radius
            
            # PATH TOLERANCE: Agent must stay reasonably close to path
            if min_distance_to_path > 8.0:
                # Always give penalty for being off path (teaches proper navigation)
                penalty_reward = -50.0
                
                # Only set path_violation (game ending) if NOT in safe zones
                if not (in_spawn_safe_zone or in_target_safe_zone):
                    self.path_violation = True
                
                return penalty_reward  # Give penalty but maybe don't end game
        
        # STUCK DETECTION - Anti-spawn-camping fix (NEW FIX)
        if self.current_step > 20:  # After initial steps
            if hasattr(self, 'position_history'):
                # Check if agent hasn't moved much in last 8 steps
                recent_positions = self.position_history[-min(8, len(self.position_history)):]
                if len(recent_positions) >= 6:
                    position_variance = 0
                    for pos in recent_positions:
                        position_variance += abs(pos[0] - self.agent_x) + abs(pos[1] - self.agent_y)
                    
                    # If very little movement, penalize heavily
                    if position_variance < 10:  # Almost no movement
                        return -30.0  # Strong penalty for not exploring
        
        # Calculate basic progress reward
        prev_dist = np.sqrt((prev_x - self.target_x)**2 + (prev_y - self.target_y)**2)
        distance_change = prev_dist - current_dist
        
        # Start with base reward for staying on path
        reward = 5.0
        
        # Progress toward target
        if distance_change > 0:
            reward += distance_change * 25.0  # Reward for getting closer
        
        # # ENHANCED Anti-rotation system - prevent circling behavior
        # if self.last_action in [1, 2]:  # Rotation actions
        #     reward -= 15.0  # Stronger base penalty for any rotation
            
        #     # ESCALATING penalty for consecutive rotations
        #     if hasattr(self, 'action_history'):
        #         recent_rotations = sum(1 for a in self.action_history[:3] if a in [1, 2])
        #         if recent_rotations >= 2:
        #             reward -= 25.0 * (recent_rotations - 1)  # Much stronger escalation
                
        #         # DIRECTION BIAS PENALTY: Heavily penalize same-direction turns
        #         same_direction_count = sum(1 for a in self.action_history[:3] if a == self.last_action)
        #         if same_direction_count >= 2:  # Same turn direction multiple times
        #             reward -= 50.0 * same_direction_count  # Huge penalty for repeated same turns
                
        #         # CIRCLING DETECTION: Check for A-D-A-D or D-A-D-A patterns
        #         if len(self.action_history) >= 3:
        #             pattern = self.action_history[:3]
        #             # Check for alternating rotation patterns (classic circling)
        #             if (pattern == [1, 2, 1] or pattern == [2, 1, 2] or 
        #                 pattern == [2, 1, 2] or pattern == [2, 1, 2]):
        #                 reward -= 100.0  # Massive penalty for circling patterns
        
        # FULL ROTATION PENALTY - Penalize agent for turning around itself
        if self.total_rotation_degrees >= 360.0:
            # Calculate how many full rotations completed
            new_full_rotations = int(self.total_rotation_degrees // 360.0)
            if new_full_rotations > self.full_rotations_completed:
                # Agent completed one or more additional full rotations
                rotations_penalty = (new_full_rotations - self.full_rotations_completed) * 200.0
                reward -= rotations_penalty
                self.full_rotations_completed = new_full_rotations
                # Optional: reset rotation counter to prevent runaway accumulation
                # self.total_rotation_degrees = self.total_rotation_degrees % 360.0
        
        # Forward movement bonus
        if self.last_action == 0 and not self.collision_occurred:  # Successful forward
            reward += 5.0
        
        # SMART Collision penalty - reduce fear near target
        if self.collision_occurred:
            # Reduce collision penalty when very close to target (encourage final approach)
            if current_dist <= 8.0:  # Very close to target
                reward -= 3.0  # Much smaller penalty near target
            elif current_dist <= 15.0:  # Close to target
                reward -= 6.0  # Reduced penalty
            else:
                reward -= 10.0  # Normal penalty when far from target
        
        # COURAGE BONUS: Reward forward attempts near target even if they fail
        if self.last_action == 0 and current_dist <= 10.0:  # Forward attempt near target
            reward += 2.0  # Small bonus for trying forward movement near target
        
        return reward
    
    
    def _point_to_line_distance(self, point, line_start, line_end):
        """Calculate shortest distance from point to line segment"""
        # Vector from line start to line end
        line_vec = line_end - line_start
        # Vector from line start to point
        point_vec = point - line_start
        
        # If line has zero length, return distance to start point
        line_len_sq = np.dot(line_vec, line_vec)
        if line_len_sq == 0:
            return np.linalg.norm(point_vec)
        
        # Project point onto line
        t = max(0, min(1, np.dot(point_vec, line_vec) / line_len_sq))
        
        # Find closest point on line segment
        closest_point = line_start + t * line_vec
        
        # Return distance to closest point
        return np.linalg.norm(point - closest_point)
    
    
    
    
    
    def _check_all_obstacles_collision(self, x, y):
        """Check if position collides with ANY of the obstacles"""
        point = np.array([x, y])
        
        for obstacle in self.obstacles:
            # Handle circle obstacles
            if obstacle.get('type') == 'circle':
                center = np.array(obstacle['center'])
                radius = obstacle['radius']
                distance = np.linalg.norm(point - center)
                if distance <= radius:
                    return True
            
            # Handle boundary wall obstacles (vertical walls)
            elif obstacle.get('type') == 'boundary_wall':
                wall_x = obstacle['x_position']
                wall_y_start = obstacle['y_start']
                wall_y_end = obstacle['y_end']
                thickness = obstacle['thickness']
                
                # Check if point is within Y range of wall and close enough in X
                if wall_y_start <= y <= wall_y_end:
                    # For x=762 boundary wall, STRICT prevention of crossing
                    # Agent should NEVER be allowed at x < 762
                    if wall_x == 762:
                        if x < wall_x:  # Agent trying to be left of x=762 (forbidden zone)
                            return True
                    else:
                        # For other boundary walls, use normal thickness check
                        distance_to_wall = abs(x - wall_x)
                        if distance_to_wall <= thickness / 2:
                            return True
            
            # Handle line obstacles (original logic)
            else:
                obstacle_start = np.array(obstacle['start'])
                obstacle_end = np.array(obstacle['end'])
                thickness = obstacle['thickness']
                
                distance = self._point_to_line_distance(point, obstacle_start, obstacle_end)
                
                # If collision with this obstacle, return True
                if distance <= thickness:
                    return True
        
        # No collision with any obstacle
        return False
    
    def _is_direction_blocked(self, direction_angle):
        """Check if movement in a specific direction (angle) is blocked by obstacles
        direction_angle: 0=forward, 90=left, 270=right from current agent angle"""
        
        # Calculate the actual world direction
        actual_angle = (self.agent_angle + direction_angle) % 360
        angle_rad = np.radians(actual_angle)
        
        # Check positions at different distances to detect obstacles
        test_distance = 3  # Check 3 pixels ahead
        new_x = self.agent_x + test_distance * np.cos(angle_rad)  
        new_y = self.agent_y + test_distance * np.sin(angle_rad)
        
        # Check world boundaries
        if new_x < self.world_x_min or new_x > self.world_x_max or new_y < self.world_y_min or new_y > self.world_y_max:
            return 1.0  # Blocked by boundary
            
        # Check obstacles
        if self._check_all_obstacles_collision(int(new_x), int(new_y)):
            return 1.0  # Blocked by obstacle
            
        return 0.0  # Not blocked
    
    
    
    
    
    def _distance_to_target(self):
        """Distance to target"""
        dx = self.agent_x - self.target_x
        dy = self.agent_y - self.target_y
        return np.sqrt(dx * dx + dy * dy)
    
    def _target_reached(self):
        """Check if target reached"""
        return self._distance_to_target() < self.target_radius
    
    def _is_done(self):
        """Check if episode done"""
        if self._target_reached():
            return True
        if self.current_step >= self.max_steps:
            return True
        # Check if agent went too far from path
        if hasattr(self, 'path_violation') and self.path_violation:
            return True
        return False
    
    def render(self):
        """Render environment"""
        if not self.render_mode:
            return
            
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Clear screen
        self.window.fill((240, 245, 250))
        
        # Convert world coordinates to screen coordinates
        # IMPORTANT: Flip Y-axis for screen display to match real game coordinate system
        # Real game: Y increases upward, Screen: Y increases downward
        def world_to_screen(wx, wy):
            sx = (wx - self.world_x_min) / self.world_width * self.window_size
            sy = self.window_size - (wy - self.world_y_min) / self.world_height * self.window_size
            return int(sx), int(sy)
        
        # Draw the path as connected lines
        if len(self.real_game_path) > 1:
            path_points = []
            for point in self.real_game_path:
                screen_point = world_to_screen(point[0], point[1])
                path_points.append(screen_point)
            
            # Draw critical path boundary (6.5 units from path)
            critical_distance = 15
            for i in range(len(path_points) - 1):
                # Draw thicker line showing critical boundary
                pygame.draw.line(self.window, (255, 100, 100), path_points[i], path_points[i + 1], 
                               max(1, int(critical_distance * 2 / max(self.world_width, self.world_height) * self.window_size)))
            
            # Draw path as connected lines (green line like training mode)
            for i in range(len(path_points) - 1):
                pygame.draw.line(self.window, (0, 200, 0), path_points[i], path_points[i + 1], 3)
            
            # Draw only start and end points (no intermediate points)
            if len(path_points) > 0:
                # Starting point - green circle
                pygame.draw.circle(self.window, (0, 255, 0), path_points[0], 8)
                # End point - red circle (target)
                pygame.draw.circle(self.window, (255, 0, 0), path_points[-1], 8)
        
        # Draw target
        target_screen = world_to_screen(self.target_x, self.target_y)
        target_radius_screen = int(self.target_radius / max(self.world_width, self.world_height) * self.window_size)
        pygame.draw.circle(self.window, (255, 50, 50), target_screen, target_radius_screen)
        
        # Draw safe zones - green circles around spawn and target
        safe_radius = 20.0  # Safe zone radius
        safe_radius_screen = int(safe_radius / max(self.world_width, self.world_height) * self.window_size)
        
        # Spawn safe zone - light green circle
        spawn_screen = world_to_screen(self.initial_location[0], self.initial_location[1])
        pygame.draw.circle(self.window, (100, 255, 100), spawn_screen, safe_radius_screen, 2)  # Light green outline
        
        # Target safe zone - light green circle  
        target_safe_screen = world_to_screen(self.target_location[0], self.target_location[1])
        pygame.draw.circle(self.window, (100, 255, 100), target_safe_screen, safe_radius_screen, 2)  # Light green outline
        
        # Draw all 8 obstacles creating corridors
        obstacle_colors = [(139, 69, 19), (180, 82, 45), (160, 82, 45), (101, 67, 33), (92, 51, 23), (139, 90, 43), (160, 100, 50)]  # Brown, Saddle brown, Dark brown, Dark saddle brown, Very dark brown, Light brown, Medium brown
        
        for i, obstacle in enumerate(self.obstacles):
            color = obstacle_colors[i] if i < len(obstacle_colors) else (139, 69, 19)
            
            # Draw circle obstacles
            if obstacle.get('type') == 'circle':
                center_screen = world_to_screen(obstacle['center'][0], obstacle['center'][1])
                radius_screen = max(1, int(obstacle['radius'] * 4))  # Scale radius for visibility
                pygame.draw.circle(self.window, color, center_screen, radius_screen)
            
            # Draw boundary wall obstacles
            elif obstacle.get('type') == 'boundary_wall':
                wall_start_screen = world_to_screen(obstacle['x_position'], obstacle['y_start'])
                wall_end_screen = world_to_screen(obstacle['x_position'], obstacle['y_end'])
                thickness_screen = max(2, int(obstacle['thickness'] * 3))  # Make boundary wall more visible
                pygame.draw.line(self.window, (200, 0, 0), wall_start_screen, wall_end_screen, thickness_screen)
            
            # Draw line obstacles (original logic)
            else:
                obstacle_start_screen = world_to_screen(obstacle['start'][0], obstacle['start'][1])
                obstacle_end_screen = world_to_screen(obstacle['end'][0], obstacle['end'][1])
                thickness_screen = max(1, int(obstacle['thickness'] * 2))
                pygame.draw.line(self.window, color, obstacle_start_screen, obstacle_end_screen, thickness_screen)
        
        # Draw corridor boundaries for visualization
        if len(self.obstacles) >= 3:
            # Left corridor center
            left_center_x = (self.left_corridor_x_min + self.left_corridor_x_max) / 2
            left_center_y = (self.corridor_y_min + self.corridor_y_max) / 2
            left_center_screen = world_to_screen(left_center_x, left_center_y)
            
            # Right corridor center
            right_center_x = (self.right_corridor_x_min + self.right_corridor_x_max) / 2
            right_center_y = (self.corridor_y_min + self.corridor_y_max) / 2
            right_center_screen = world_to_screen(right_center_x, right_center_y)
            
            # Draw corridor markers
            pygame.draw.circle(self.window, (0, 255, 0), left_center_screen, 3)   # Green - Left corridor
            pygame.draw.circle(self.window, (0, 255, 255), right_center_screen, 3) # Cyan - Right corridor
        
        
        # Draw agent
        agent_screen = world_to_screen(self.agent_x, self.agent_y)
        pygame.draw.circle(self.window, (0, 120, 255), agent_screen, self.agent_display_radius)
        
        # Draw agent direction arrow (corrected for flipped Y-axis)
        # For screen display, we need to flip the angle to maintain visual consistency
        # Since screen Y is flipped, we negate the angle to maintain left/right visual consistency
        display_angle_rad = np.radians(-self.agent_angle)
        arrow_length = 20
        arrow_end = (
            agent_screen[0] + arrow_length * np.cos(display_angle_rad),
            agent_screen[1] + arrow_length * np.sin(display_angle_rad)
        )
        pygame.draw.line(self.window, (255, 0, 0), agent_screen, arrow_end, 3)
        
        # Draw info
        distance = self._distance_to_target()
        
        # Calculate distance to path for display
        min_dist_to_path = float('inf')
        if len(self.real_game_path) >= 2:
            agent_pos = np.array([self.agent_x, self.agent_y])
            for i in range(len(self.real_game_path) - 1):
                point1 = np.array(self.real_game_path[i])
                point2 = np.array(self.real_game_path[i + 1])
                dist = self._point_to_line_distance(agent_pos, point1, point2)
                if dist < min_dist_to_path:
                    min_dist_to_path = dist
        
        # Show path violation status
        path_status = "VIOLATION!" if hasattr(self, 'path_violation') and self.path_violation else "OK"
        path_color = (255, 0, 0) if hasattr(self, 'path_violation') and self.path_violation else (0, 150, 0)
        
        info_text = f"Step: {self.current_step}/{self.max_steps} | Distance: {distance:.1f} | Path: {min_dist_to_path:.1f}/9"
        text_surface = self.font.render(info_text, True, (0, 0, 0))
        self.window.blit(text_surface, (10, 10))
        
        path_text = f"Path Status: {path_status}"
        path_surface = self.font.render(path_text, True, path_color)
        self.window.blit(path_surface, (10, 35))
        
        # Show state vector in manual play (split into multiple lines for readability)
        if hasattr(self, 'last_state') and self.last_state is not None:
            # First line: Distance, velocity, actions
            state_text1 = f"State: Dist:{self.last_state[0]:.2f} Vel:({self.last_state[1]:.2f},{self.last_state[2]:.2f}) Act:[{self.last_state[3]:.2f},{self.last_state[4]:.2f},{self.last_state[5]:.2f}]"
            state_surface1 = self.font.render(state_text1, True, (0, 0, 100))
            self.window.blit(state_surface1, (10, 60))
            
            # Second line: dx and dy histories
            state_text2 = f"       dx:[{self.last_state[6]:.2f},{self.last_state[7]:.2f},{self.last_state[8]:.2f}] dy:[{self.last_state[9]:.2f},{self.last_state[10]:.2f},{self.last_state[11]:.2f}]"
            state_surface2 = self.font.render(state_text2, True, (0, 0, 100))
            self.window.blit(state_surface2, (10, 85))
        
        # Show last reward in manual play
        if hasattr(self, 'last_reward_display'):
            reward_text = f"Last Reward: {self.last_reward_display:.1f}"
            reward_color = (0, 150, 0) if self.last_reward_display > 0 else (150, 0, 0)
            reward_surface = self.font.render(reward_text, True, reward_color)
            self.window.blit(reward_surface, (10, 110))
        
        # Show agent position in manual play
        position_text = f"Agent Position: ({self.agent_x:.0f}, {self.agent_y:.0f})"
        position_surface = self.font.render(position_text, True, (0, 100, 0))
        self.window.blit(position_surface, (10, 135))
        
        # Show target position in manual play
        target_text = f"Target Position: ({self.target_x:.0f}, {self.target_y:.0f})"
        target_surface = self.font.render(target_text, True, (100, 0, 100))
        self.window.blit(target_surface, (10, 160))
        
        # Show current path info in manual play
        if hasattr(self, 'real_game_path') and self.real_game_path:
            path_info = f"Path: {self.real_game_path[0]} -> {self.real_game_path[-1]} ({len(self.real_game_path)} waypoints)"
            path_surface = self.font.render(path_info, True, (0, 0, 150))
            self.window.blit(path_surface, (10, 185))
        
        
        # Add boundary warning text
        if self.agent_x < 762:
            warning_text = "WARNING: Agent beyond X=762 boundary!"
            warning_surface = self.font.render(warning_text, True, (255, 0, 0))
            self.window.blit(warning_surface, (10, 210))
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
    
    def render_debug(self, reward=None):
        """Enhanced render with state and reward information"""
        if not self.render_mode:
            return
            
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        # Clear screen
        self.window.fill((240, 245, 250))
        
        # Convert world coordinates to screen coordinates
        # IMPORTANT: Flip Y-axis for screen display to match real game coordinate system
        # Real game: Y increases upward, Screen: Y increases downward
        def world_to_screen(wx, wy):
            sx = (wx - self.world_x_min) / self.world_width * self.window_size
            sy = self.window_size - (wy - self.world_y_min) / self.world_height * self.window_size
            return int(sx), int(sy)
        
        # Draw the path as connected lines
        if len(self.real_game_path) > 1:
            path_points = []
            for point in self.real_game_path:
                screen_point = world_to_screen(point[0], point[1])
                path_points.append(screen_point)
            
            # Draw critical path boundary (6.5 units from path)
            critical_distance = 6.5
            for i in range(len(path_points) - 1):
                # Draw thicker line showing critical boundary
                pygame.draw.line(self.window, (255, 100, 100), path_points[i], path_points[i + 1], 
                               max(1, int(critical_distance * 2 / max(self.world_width, self.world_height) * self.window_size)))
            
            # Draw path as connected lines (green line like training mode)
            for i in range(len(path_points) - 1):
                pygame.draw.line(self.window, (0, 200, 0), path_points[i], path_points[i + 1], 3)
            
            # Draw only start and end points (no intermediate points)
            if len(path_points) > 0:
                # Starting point - green circle
                pygame.draw.circle(self.window, (0, 255, 0), path_points[0], 8)
                # End point - red circle (target)
                pygame.draw.circle(self.window, (255, 0, 0), path_points[-1], 8)
        
        # Draw target
        target_screen = world_to_screen(self.target_x, self.target_y)
        target_radius_screen = int(self.target_radius / max(self.world_width, self.world_height) * self.window_size)
        pygame.draw.circle(self.window, (255, 50, 50), target_screen, target_radius_screen)
        
        # Draw safe zones - green circles around spawn and target
        safe_radius = 10.0  # Safe zone radius
        safe_radius_screen = int(safe_radius / max(self.world_width, self.world_height) * self.window_size)
        
        # Spawn safe zone - light green circle
        spawn_screen = world_to_screen(self.initial_location[0], self.initial_location[1])
        pygame.draw.circle(self.window, (100, 255, 100), spawn_screen, safe_radius_screen, 2)  # Light green outline
        
        # Target safe zone - light green circle  
        target_safe_screen = world_to_screen(self.target_location[0], self.target_location[1])
        pygame.draw.circle(self.window, (100, 255, 100), target_safe_screen, safe_radius_screen, 2)  # Light green outline
        
        # Draw all 8 obstacles creating corridors
        obstacle_colors = [(139, 69, 19), (180, 82, 45), (160, 82, 45), (101, 67, 33), (92, 51, 23), (139, 90, 43), (160, 100, 50)]  # Brown, Saddle brown, Dark brown, Dark saddle brown, Very dark brown, Light brown, Medium brown
        
        for i, obstacle in enumerate(self.obstacles):
            color = obstacle_colors[i] if i < len(obstacle_colors) else (139, 69, 19)
            
            # Draw circle obstacles
            if obstacle.get('type') == 'circle':
                center_screen = world_to_screen(obstacle['center'][0], obstacle['center'][1])
                radius_screen = max(1, int(obstacle['radius'] * 4))  # Scale radius for visibility
                pygame.draw.circle(self.window, color, center_screen, radius_screen)
            
            # Draw boundary wall obstacles
            elif obstacle.get('type') == 'boundary_wall':
                wall_start_screen = world_to_screen(obstacle['x_position'], obstacle['y_start'])
                wall_end_screen = world_to_screen(obstacle['x_position'], obstacle['y_end'])
                thickness_screen = max(2, int(obstacle['thickness'] * 3))  # Make boundary wall more visible
                pygame.draw.line(self.window, (200, 0, 0), wall_start_screen, wall_end_screen, thickness_screen)
            
            # Draw line obstacles (original logic)
            else:
                obstacle_start_screen = world_to_screen(obstacle['start'][0], obstacle['start'][1])
                obstacle_end_screen = world_to_screen(obstacle['end'][0], obstacle['end'][1])
                thickness_screen = max(1, int(obstacle['thickness'] * 2))
                pygame.draw.line(self.window, color, obstacle_start_screen, obstacle_end_screen, thickness_screen)
        
        # Draw corridor boundaries for visualization
        if len(self.obstacles) >= 3:
            # Left corridor center
            left_center_x = (self.left_corridor_x_min + self.left_corridor_x_max) / 2
            left_center_y = (self.corridor_y_min + self.corridor_y_max) / 2
            left_center_screen = world_to_screen(left_center_x, left_center_y)
            
            # Right corridor center
            right_center_x = (self.right_corridor_x_min + self.right_corridor_x_max) / 2
            right_center_y = (self.corridor_y_min + self.corridor_y_max) / 2
            right_center_screen = world_to_screen(right_center_x, right_center_y)
            
            # Draw corridor markers
            pygame.draw.circle(self.window, (0, 255, 0), left_center_screen, 3)   # Green - Left corridor
            pygame.draw.circle(self.window, (0, 255, 255), right_center_screen, 3) # Cyan - Right corridor
        
        
        # Draw agent
        agent_screen = world_to_screen(self.agent_x, self.agent_y)
        pygame.draw.circle(self.window, (0, 120, 255), agent_screen, self.agent_display_radius)
        
        # Draw agent direction arrow (corrected for flipped Y-axis)
        # For screen display, we need to flip the angle to maintain visual consistency
        # Since screen Y is flipped, we negate the angle to maintain left/right visual consistency
        display_angle_rad = np.radians(-self.agent_angle)
        arrow_length = 20
        arrow_end = (
            agent_screen[0] + arrow_length * np.cos(display_angle_rad),
            agent_screen[1] + arrow_length * np.sin(display_angle_rad)
        )
        pygame.draw.line(self.window, (255, 0, 0), agent_screen, arrow_end, 3)
        
        # Draw detailed info
        distance = self._distance_to_target()
        state = self._get_state()
        
        # Calculate raw dx/dy values
        raw_dx = self.target_x - self.agent_x
        raw_dy = self.target_y - self.agent_y
        
        # Basic info
        info_lines = [
            f"Step: {self.current_step}/{self.max_steps} | Distance: {distance:.1f}",
            f"Agent: ({self.agent_x:.1f}, {self.agent_y:.1f}) | Target: ({self.target_x}, {self.target_y})",
            f"Raw dx: {raw_dx:.1f}, Raw dy: {raw_dy:.1f}",
            ""
        ]
        
        # State information
        state_labels = ["dist_norm", "vel_x", "vel_y", "act_0", "act_1", "act_2", 
                       "dx_curr", "dx_t-1", "dx_t-2", "dy_curr", "dy_t-1", "dy_t-2"]
        info_lines.append("STATE VECTOR (12D normalized):")
        for i, (label, value) in enumerate(zip(state_labels, state)):
            info_lines.append(f"  {label}: {value:.3f}")
        
        # Reward information
        if reward is not None:
            info_lines.append("")
            info_lines.append(f"REWARD: {reward:.2f}")
        
        # Action history with better formatting
        info_lines.append("")
        if hasattr(self, 'action_history'):
            action_names = ["W", "A", "S", "D", "NONE"]
            actions_str = []
            for i, act in enumerate(self.action_history):
                if act >= 0 and act < 4:
                    actions_str.append(action_names[act])
                else:
                    actions_str.append("-")
            info_lines.append(f"Action History: {actions_str[0]} -> {actions_str[1]} -> {actions_str[2]}")
            info_lines.append(f"  (Current -> Previous -> Previous-1)")
        
        # Velocity information
        if hasattr(self, 'prev_x') and hasattr(self, 'prev_y'):
            vel_x = self.agent_x - self.prev_x
            vel_y = self.agent_y - self.prev_y
            info_lines.append(f"Velocity: ({vel_x:.2f}, {vel_y:.2f})")
        
        # Collision information
        if hasattr(self, 'collision_occurred'):
            collision_status = "YES" if self.collision_occurred else "NO"
            info_lines.append(f"Edge Collision: {collision_status}")
        
        # DX/DY History
        if hasattr(self, 'dx_history') and hasattr(self, 'dy_history'):
            info_lines.append(f"DX History: {self.dx_history[0]:.1f} -> {self.dx_history[1]:.1f} -> {self.dx_history[2]:.1f}")
            info_lines.append(f"DY History: {self.dy_history[0]:.1f} -> {self.dy_history[1]:.1f} -> {self.dy_history[2]:.1f}")
        
        # Draw all info lines
        y_offset = 10
        for line in info_lines:
            if line:  # Skip empty lines
                text_surface = self.font.render(line, True, (0, 0, 0))
                self.window.blit(text_surface, (10, y_offset))
            y_offset += 18  # Reduced spacing for more info
        
        # Update display
        pygame.display.flip()
        self.clock.tick(60)
    
    def close(self):
        """Close environment"""
        if self.render_mode:
            pygame.quit()

def manual_test():
    """Manual test with random path selection each episode"""
    import random
    
    # All available paths for random selection
    all_paths = {
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
            (762, 648)   # End: Inn
        ],
        "inn_to_armor": [
            (762, 648),  # Start: Inn
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
    
    # Select random path for first episode
    path_names = list(all_paths.keys())
    current_path_name = random.choice(path_names)
    current_path = all_paths[current_path_name]
    
    env = UltraSimpleEnvironment(render_mode=True, real_game_path=current_path)
    
    print("🎮 MANUAL TEST WITH RANDOM PATHS")
    print("Available paths:", list(all_paths.keys()))
    print(f"🗺️ Starting with: {current_path_name}")
    print(f"   {current_path[0]} → {current_path[-1]}")
    print("\nControls:")
    print("W - Forward, A - Turn Left, D - Turn Right") 
    print("R - Reset environment (NEW RANDOM PATH)")
    print("ESC - Exit")
    print("🎲 Each reset will select a different random path!")
    
    running = True
    last_reward = 0.0
    total_reward = 0.0
    
    while running:
        env.render()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_w:
                    state, reward, done, info = env.step(0)
                    last_reward = reward
                    total_reward += reward
                    # Store for display
                    env.last_state = state
                    env.last_reward_display = reward
                    print(f"Forward: reward={reward:.2f}, total={total_reward:.2f}")
                    print(f"State (12D): {state}")
                    print(f"  Distance: {state[0]:.3f}, Velocity: ({state[1]:.3f}, {state[2]:.3f})")
                    print(f"  Actions: [{state[3]:.3f}, {state[4]:.3f}, {state[5]:.3f}]")
                    print(f"  dx: [{state[6]:.3f}, {state[7]:.3f}, {state[8]:.3f}]")  
                    print(f"  dy: [{state[9]:.3f}, {state[10]:.3f}, {state[11]:.3f}]")
                elif event.key == pygame.K_a:
                    state, reward, done, info = env.step(1)
                    last_reward = reward
                    total_reward += reward
                    env.last_state = state
                    env.last_reward_display = reward
                    print(f"Turn Left: reward={reward:.2f}, total={total_reward:.2f}")
                    print(f"State (12D): {state}")
                elif event.key == pygame.K_d:
                    state, reward, done, info = env.step(2)
                    last_reward = reward
                    total_reward += reward
                    env.last_state = state
                    env.last_reward_display = reward
                    print(f"Turn Right: reward={reward:.2f}, total={total_reward:.2f}")
                    print(f"State (12D): {state}")
                elif event.key == pygame.K_r:
                    # Select new random path
                    current_path_name = random.choice(path_names)
                    current_path = all_paths[current_path_name]
                    
                    # Create new environment with new path
                    env.close()
                    env = UltraSimpleEnvironment(render_mode=True, real_game_path=current_path)
                    
                    last_reward = 0.0
                    total_reward = 0.0
                    print(f"🎲 NEW RANDOM PATH: {current_path_name}")
                    print(f"   {current_path[0]} → {current_path[-1]}")
                
                if done:
                    if info['target_reached']:
                        print(f"SUCCESS! Target reached! Total reward: {total_reward:.2f}")
                    else:
                        print(f"Failed - time limit reached. Total reward: {total_reward:.2f}")
                    
                    # Select new random path for next episode
                    current_path_name = random.choice(path_names)
                    current_path = all_paths[current_path_name]
                    
                    # Create new environment with new path
                    env.close()
                    env = UltraSimpleEnvironment(render_mode=True, real_game_path=current_path)
                    
                    last_reward = 0.0
                    total_reward = 0.0
                    print(f"🎲 NEW EPISODE - RANDOM PATH: {current_path_name}")
                    print(f"   {current_path[0]} → {current_path[-1]}")
    
    env.close()

if __name__ == '__main__':
    manual_test()