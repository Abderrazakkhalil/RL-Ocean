import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces

class GW(gym.Env):
    """
    Modular GridWorld environment:
    - Configurable rectangular grid (width x height)
    - Actions: 0=up, 1=down, 2=right, 3=left
    - Configurable terminal states
    - Option to display the agent
    """

    def __init__(self, 
                 grid_width=4,
                 grid_height=4,
                 grid_size=None,  
                 terminal_states=None, # default [(0,0), (width-1,height-1)]
                 initial_state=None, # default
                 max_steps=40,
                 show_agent=True,
                 show_goals=True,
                 goal_reward=10,
                 step_penalty=-1,
                 wall_penalty=-2,
                 obstacle_penalty=-5,  # NEW: distinct penalty for obstacles
             
                 # New goal parameters (backward compatible with terminal_states)
                 num_goals=None,
                 goal_positions=None,
                 # New obstacle parameters
                 num_obstacles=0,
                 obstacle_positions=None,
                 obstacles_move=False):
        super().__init__()
        
        # Grid configuration (rectangular)
        if grid_size is not None:
            # Backward compatibility: grid_size defines a square grid
            self.grid_width = grid_size
            self.grid_height = grid_size
        else:
            self.grid_width = grid_width
            self.grid_height = grid_height
            
        self.action_space = spaces.Discrete(4)  # 4 actions
        self.observation_space = spaces.MultiDiscrete([self.grid_width, self.grid_height])
        
        # Terminal states / goals
        # - priority: goal_positions > terminal_states > defaults
        self._init_terminal_states = terminal_states
        self._init_goal_positions = goal_positions
        self._init_num_goals = num_goals
        # Defaults if nothing is provided (opposite corners)
        self.terminal_states = [(0, 0), (self.grid_width-1, self.grid_height-1)]
            
        # Initial state
        self.initial_state = initial_state
        
        # Reward parameters
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        self.wall_penalty = wall_penalty
        self.obstacle_penalty = obstacle_penalty  # NEW
        
        # Display parameters
        self.show_agent = show_agent
        self.show_goals = show_goals
       
        # Obstacles
        self._init_num_obstacles = int(num_obstacles) if num_obstacles is not None else 0
        self._init_obstacle_positions = obstacle_positions
        self.obstacles_move = bool(obstacles_move)
        self.obstacles = []  # liste de positions (x, y)
        
        # Environment state
        self.state = np.array([0, 0])
        self.terminated = False
        self.truncated = False
        self.step_counter = 0
        self.max_steps = max_steps

        # For display
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate goals
        # Determine the self.terminal_states list based on init parameters
        if self._init_goal_positions is not None:
            goals = [tuple(map(int, p)) for p in self._init_goal_positions]
        elif self._init_terminal_states is not None:
            goals = [tuple(map(int, p)) for p in self._init_terminal_states]
        else:
            # Random sampling if num_goals is specified, otherwise default corners
            if self._init_num_goals is None:
                goals = [(0, 0), (self.grid_width - 1, self.grid_height - 1)]
            else:
                num_goals = int(self._init_num_goals)
                num_goals = max(1, min(num_goals, self.grid_width * self.grid_height - 1))
                all_cells = [(x, y) for x in range(self.grid_width) for y in range(self.grid_height)]
                # sample without condition on the agent for now (agent will be placed later)
                idx = self.np_random.choice(len(all_cells), size=num_goals, replace=False)
                goals = [all_cells[i] for i in np.atleast_1d(idx)]
        self.terminal_states = goals

        # Initial position of the agent (avoid goals and future obstacles if via fixed positions)
        if self.initial_state is not None:
            self.state = np.array(self.initial_state, dtype=int)
        else:
            while True:
                x = self.np_random.integers(0, self.grid_width)
                y = self.np_random.integers(0, self.grid_height)
                if (x, y) not in self.terminal_states:
                    self.state = np.array([x, y], dtype=int)
                    break

        # Generate obstacles
        if self._init_obstacle_positions is not None:
            obstacles = [tuple(map(int, p)) for p in self._init_obstacle_positions]
        else:
            obstacles = []
            count = max(0, min(self._init_num_obstacles, self.grid_width * self.grid_height - len(self.terminal_states) - 1))
            if count > 0:
                forbidden = set(self.terminal_states + [tuple(self.state)])
                free_cells = [(x, y) for x in range(self.grid_width) for y in range(self.grid_height) if (x, y) not in forbidden]
                if len(free_cells) >= count:
                    idx = self.np_random.choice(len(free_cells), size=count, replace=False)
                    obstacles = [free_cells[i] for i in np.atleast_1d(idx)]
        self.obstacles = obstacles

        # Reset
        self.step_counter = 0
        self.terminated = False
        self.truncated = False

        info = {}
        return self.state, info

    def step(self, action):
        """
        Move in the grid according to the action
        """
        if self.terminated or self.truncated:
            raise RuntimeError("L'environnement est terminé. Appelle reset().")

        self.step_counter += 1
        x, y = self.state.copy()

        # Define moves with border handling
        if action == 0:   # Up
           y = max(0, y - 1)
        elif action == 1:  # Down
            y = min(self.grid_height-1, y + 1)
        elif action == 2:  # Right
            x = min(self.grid_width-1, x + 1)
        elif action == 3:  # Left
            x = max(0, x - 1)

        # Check if the target cell is an obstacle
        target = (x, y)
        if target in self.obstacles:
            # Do not move, apply obstacle penalty (MODIFIED)
            reward = self.obstacle_penalty
            moved = False
            new_state = self.state.copy()  # Reste à la même position
        else:
            # Normal move
            new_state = np.array([x, y])
            moved = not np.array_equal(new_state, self.state)
            # Check whether the new state is a terminal state
            if tuple(new_state) in self.terminal_states:
                self.terminated = True
                reward = self.goal_reward
            elif moved:
                reward = self.step_penalty
            else:
                reward = self.wall_penalty  # penalty if blocked by borders
        
        self.state = new_state

        # Move obstacles (random 4-connected) if enabled
        if not self.terminated and self.obstacles_move and len(self.obstacles) > 0:
            new_obstacles = []
            # Forbidden cells: goals and agent position
            forbidden_after = set(self.terminal_states + [tuple(self.state)])
            for ox, oy in self.obstacles:
                candidates = [
                    (min(self.grid_width - 1, ox + 1), oy),
                    (max(0, ox - 1), oy),
                    (ox, min(self.grid_height - 1, oy + 1)),
                    (ox, max(0, oy - 1)),
                    (ox, oy),  # possibilité de rester
                ]
                # Avoid collisions between obstacles by choosing sequentially
                self.np_random.shuffle(candidates)
                chosen = None
                for cx, cy in candidates:
                    if (cx, cy) in forbidden_after:
                        continue
                    if (cx, cy) in new_obstacles:
                        continue
                    chosen = (cx, cy)
                    break
                if chosen is None:
                    chosen = (ox, oy)
                new_obstacles.append(chosen)
            self.obstacles = new_obstacles

        # End if too long
        if self.step_counter >= self.max_steps:
            self.truncated = True

        return self.state, reward, self.terminated, self.truncated, {}

    def render(self, mode="human"):
        """Graphical display of the grid and the agent's position"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
            plt.ion()

        self.ax.clear()
        
        # Limits based on grid dimensions
        self.ax.set_xlim(-0.5, self.grid_width - 0.5)
        self.ax.set_ylim(-0.5, self.grid_height - 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.grid(True, alpha=0.5)

        # Draw cells
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Centered coordinates with vertical inversion for display
                y_display = self.grid_height - 1 - y
                
                color = "lightblue"
                if (x, y) in self.terminal_states:
                    color = "lightcoral"  # goals in light red
                if (x, y) in self.obstacles:
                    color = "gray"

                rect = plt.Rectangle(
                    (x - 0.4, y_display - 0.4),
                    0.8, 0.8,
                    facecolor=color, edgecolor='black', linewidth=2, alpha=0.7)
                self.ax.add_patch(rect)

                # Mark goals
                if self.show_goals and (x, y) in self.terminal_states:
                    self.ax.text(x, y_display, "GOAL", ha='center', va='center', 
                            fontsize=9, fontweight='bold', color='darkred')

        # Agent
        if self.show_agent:
            x, y = self.state
            y_display = self.grid_height - 1 - y
            
            radius = 0.3
            agent_circle = plt.Circle((x, y_display), radius,
                                    facecolor='green',
                                    edgecolor='black',
                                    linewidth=2)
            self.ax.add_patch(agent_circle)

            # Eyes
            self.ax.plot(x + 0.1, y_display + 0.15, 'ko', markersize=2)
            self.ax.plot(x - 0.1, y_display + 0.15, 'ko', markersize=2)
            
            # Facial expression depending on the state
            if self.terminated and (x, y) in self.terminal_states:
                # Smile (goal reached) - upward parabola
                mouth_x = np.linspace(x - 0.15, x + 0.15, 20)
                mouth_y = y_display - 0.1 + 0.05 * ((mouth_x - x) / 0.15) ** 2
                self.ax.plot(mouth_x, mouth_y, 'k-', linewidth=2)
            elif self.truncated:
                # Despair (timeout) - downward parabola
                mouth_x = np.linspace(x - 0.15, x + 0.15, 20)
                mouth_y = y_display - 0.1 - 0.05 * ((mouth_x - x) / 0.15) ** 2
                self.ax.plot(mouth_x, mouth_y, 'k-', linewidth=2)
            else:
                # Strict grimace (exploration)
                self.ax.plot([x - 0.15, x + 0.15], [y_display - 0.1, y_display - 0.1], 'k-', linewidth=3)

        self.ax.set_title(f"Position: {self.state} | Step: {self.step_counter}")
        plt.draw()
        plt.pause(0.3)

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None