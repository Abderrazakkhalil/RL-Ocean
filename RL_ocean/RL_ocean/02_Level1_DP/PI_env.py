import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the environment from the renamed package
from GridWorld.grid_env import GW

class PI(GW):
    """
    Specialized PI environment based on GW
    """
    
    def __init__(self):
        super().__init__(
            grid_width=4,
            grid_height=4,
            terminal_states=[(0, 0), (3, 3)],
            initial_state=(0, 0),
            max_steps=100,
            goal_reward=1,
            step_penalty=-1,
            wall_penalty=-2,
            obstacle_penalty=-2,
            show_agent=False,
            show_goals=True,
            num_obstacles=0  # No obstacles
        )
        
    def step_state(self, state, action):
        """
        Deterministic transition for a given state (PI-specific method)
        Compatible with your existing code
        """
        x, y = state.copy()
        pref_state = np.array([x, y])
        
        if action == 0:  # Up
            y = max(0, y - 1)
        elif action == 1:  # Down
             y = min(self.grid_height-1, y + 1)
        elif action == 2:  # Right
            x = min(self.grid_width-1, x + 1)
        elif action == 3:  # Left
            x = max(0, x - 1)

        new_state = np.array([x, y])

        # Reward according to your PI logic
        if tuple(new_state) in self.terminal_states:
            reward = 1  # goal
        elif not np.array_equal(new_state, pref_state):
            reward = -1  # movement cost
        else:
            reward = -2  # wall collision

        terminated = tuple(new_state) in self.terminal_states

        return new_state, reward, terminated, False, {}

    def render(self, V=None, policy=None, iteration=0, mode="human"):
        """Custom render for PI with value display"""
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            plt.ion()

        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_width - 0.5)
        self.ax.set_ylim(-0.5, self.grid_height - 0.5)
        self.ax.set_aspect('equal')
        
        self.ax.set_xticks(range(self.grid_width))
        self.ax.set_yticks(range(self.grid_height))
        self.ax.set_xticklabels([str(i) for i in range(self.grid_width)], fontsize=10)
        self.ax.set_yticklabels([str(self.grid_height-1-i) for i in range(self.grid_height)], fontsize=10)
        self.ax.grid(True, alpha=0.3, color='#CCCCCC')
        self.ax.set_facecolor('#FAFAFA')

        for x in range(self.grid_width):
            for y in range(self.grid_height):
                y_display = self.grid_height - 1 - y
                
                if (x, y) in self.terminal_states:
                    color = "#92BD98"
                    border_color = "#1F5024"
                    border_width = 3
                else:
                    color = "#E3F2FD"
                    border_color = "#1C4B6F"
                    border_width = 2

                rect = plt.Rectangle((x - 0.4, y_display - 0.4), 0.8, 0.8,
                                     facecolor=color, edgecolor=border_color, 
                                     linewidth=border_width, alpha=0.9)
                self.ax.add_patch(rect)
                
                # Mark goals
                if self.show_goals and (x, y) in self.terminal_states:
                    self.ax.text(x, y_display, "GOAL", ha='center', va='center', 
                            fontsize=9, fontweight='bold', color= "#1F5024")
       
                if V is not None:
                    value = V[x, y]
                    if (x, y) in self.terminal_states:
                        text_color = '#E17055'
                        font_weight = 'bold'
                    else:
                        text_color = '#2D3436'
                        font_weight = 'normal'
                    if (x, y) in self.terminal_states :
                        continue
                    else :
                        self.ax.text(x, y_display, f"{value:.2f}", ha='center', va='center', 
                               color=text_color, fontsize=12, fontweight=font_weight,
                               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', 
                                       edgecolor=border_color, linewidth=1, alpha=0.8))
                
        title = f"Policy Iteration - Iteration {iteration}"
        self.ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color='#2C3E50')
        
        plt.draw()
        plt.pause(3)
