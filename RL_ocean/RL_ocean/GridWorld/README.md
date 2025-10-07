# Grid Environment (`grid_env.py`)

## Overview
This module defines a **modular GridWorld environment** implemented using `gymnasium`.  
It is designed for Reinforcement Learning experiments, allowing flexible configurations for grid size, terminal states (goals), obstacles, and rewards.  
The environment is useful for testing classical algorithms (Policy Iteration, Q-Learning) and exploring their behavior in dynamic, structured environments.

---

## Key Features
- **Configurable Grid**: Supports rectangular grids (`grid_width` × `grid_height`) or square grids (`grid_size`).
- **Actions**:  
  - `0`: move up  
  - `1`: move down  
  - `2`: move right  
  - `3`: move left
- **Terminal states (goals)**: Customizable positions with positive rewards.
- **Obstacles**: Static or moving cells with distinct penalties.
- **Rewards**:
  - `goal_reward` when reaching a goal
  - `step_penalty` for each move
  - `wall_penalty` when blocked by the wall
  - `obstacle_penalty` when attempting to move into an obstacle
- **Visualization**: Optional graphical rendering with agent, goals, and obstacles.

---

## Class: `GW`

### Initialization
```python
GW(
    grid_width=4,
    grid_height=4,
    grid_size=None,
    terminal_states=None,# default [(0,0), (width-1,height-1)]
    initial_state=None, #default Random
    max_steps=40,
    show_agent=True,
    show_goals=True,
    goal_reward=10,
    step_penalty=-1,
    wall_penalty=-2,
    obstacle_penalty=-5,
    num_goals=None,
    goal_positions=None,
    num_obstacles=0,
    obstacle_positions=None,
    obstacles_move=False
)
````

### Parameters

| Parameter                   | Type          | Description                                                  |
| --------------------------- | ------------- | ------------------------------------------------------------ |
| `grid_width`, `grid_height` | `int`         | Define the dimensions of the rectangular grid                |
| `grid_size`                 | `int`         | Alternative to define a square grid (overrides width/height) |
| `terminal_states`           | `list[tuple]` | Fixed list of terminal (goal) positions                      |
| `initial_state`             | `tuple`       | Starting position of the agent                               |
| `max_steps`                 | `int`         | Maximum number of steps per episode                          |
| `show_agent`                | `bool`        | Display the agent in rendering                               |
| `show_goals`                | `bool`        | Display goal labels in rendering                             |
| `goal_reward`               | `float`       | Reward for reaching a goal                                   |
| `step_penalty`              | `float`       | Penalty for each step                                        |
| `wall_penalty`              | `float`       | Penalty for hitting a boundary                               |
| `obstacle_penalty`          | `float`       | Penalty for trying to move into an obstacle                  |
| `num_goals`                 | `int`         | Number of randomly generated goals                           |
| `goal_positions`            | `list[tuple]` | Custom goal positions                                        |
| `num_obstacles`             | `int`         | Number of obstacles                                          |
| `obstacle_positions`        | `list[tuple]` | Fixed obstacle positions                                     |
| `obstacles_move`            | `bool`        | If `True`, obstacles move randomly at each step              |

---

## Methods

### `reset(seed=None, options=None)`

Resets the environment to an initial state:

* Randomly or manually sets the agent’s starting position
* Generates goals and obstacles
* Returns the initial observation and an empty info dictionary

```python
state, info = env.reset()
```

**Returns:**

* `state`: the initial position `[x, y]`
* `info`: an empty dictionary

---

### `step(action)`

Performs one step in the environment.

**Arguments:**

* `action`: integer in `{0, 1, 2, 3}`

**Returns:**

```python
next_state, reward, terminated, truncated, info
```

* `next_state`: updated position `[x, y]`
* `reward`: scalar value according to movement result
* `terminated`: `True` if the agent reached a goal
* `truncated`: `True` if maximum steps exceeded
* `info`: empty dictionary

**Behavior:**

* Movement constrained by grid limits
* Penalties applied for obstacles or walls
* Optional obstacle motion each step if `obstacles_move=True`

---

### `render(mode="human")`

Displays a graphical view of the environment:

* Grid cells with different colors:

  * Light blue: normal cells
  * Light coral: goal cells
  * Gray: obstacles
* The agent is displayed as a green circle
* Facial expressions:

  * Smile: goal reached
  * Sad face: timeout
  * Straight mouth: ongoing exploration

**Usage:**

```python
env.render()
```

---

### `close()`

Closes the rendering window and cleans up resources.

```python
env.close()
```

---

## Example Usage

```python
import os
import sys

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the environment from the renamed package
from GridWorld.grid_env import GW

# Initialize environment
env = GW(grid_width=5, grid_height=5, num_goals=1, num_obstacles=3)

# Reset environment
state, info = env.reset()

done = False
while not done:
    # Sample random action
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    
    # Render environment
    env.render()
    done = terminated or truncated

# Close environment
env.close()
```

---

## Notes

* This environment is **fully compatible with Gymnasium** and can be integrated into RL pipelines.
* Designed to demonstrate:

  * **Dynamic Programming** algorithms when model is known
  * **Model-free** methods like **Q-Learning** when transitions are unknown
  * **Deep RL** extensions when using high-dimensional states or complex features

So we use this environment across levels to progressively explore Reinforcement Learning concepts.

```
```
