# Q-Learning for GridWorld - Comprehensive Study

## Project Overview

This project implements and analyzes the Q-learning algorithm, a reinforcement learning method, for solving sequential decision problems in customizable GridWorld environments. The study explores various aspects of Q-learning performance through systematic experiments.

## Project Structure

```
RL_ocean/
├── 03_Level2_TD/
│   ├── q_learning.py                          # Basic Q-learning implementation
│   ├── grid_size_impact_analysis.py           # Grid size influence study
│   ├── hyperparameter_sensitivity_analysis.py # Parameter tuning analysis
│   ├── dynamic_goal_qlearning.py              # Dynamic environment study
│   └── results/
│       ├── q_learning_results/
│       │   ├── training_episode_*.gif
│       │   ├── learning_curves.png
│       │   └── optimal_policy_q_learning.png
│       ├── grid_size_comparison.png
│       ├── hyperparameter_*.png
│       └── dynamic_goal_results/
│           ├── dynamic_goal_performance.png
│           └── training_episode_*.gif
└── GridWorld/
    └── grid_env.py
```

## Core Features

### Q-Learning Algorithm
- Implementation with epsilon-greedy policy
- Exploration/exploitation balance with automatic epsilon decay
- Q-value updates: Q(s,a) ← Q(s,a) + α[r + γ·maxₐ'Q(s',a') - Q(s,a)]

### GridWorld Environment
- Configurable grid dimensions
- Customizable terminal states and obstacles
- Action space: up (0), down (1), right (2), left (3)
- Reward system with penalties and goals

### Visualization Capabilities
- Training GIFs showing agent evolution
- Learning curves (rewards and episode lengths)
- Optimal policy visualization
- Q-value and value function displays

## Dependencies

- Python 3.7+
- NumPy
- Matplotlib
- Pillow (PIL)

## Installation

```bash
pip install numpy matplotlib pillow
```

## Experiments

### 1. Basic Q-Learning Implementation

**File:** `q_learning.py`

**Purpose:** Core Q-learning implementation with static environment

**Default Configuration:**
- Grid: 8x4
- Terminal states: (0,0), (7,3)
- Obstacles: Predefined complex path
- Rewards: Goal (+10), Step penalty (-0.1), Obstacle penalty (-2)
- Learning: 1000 episodes, γ=0.9, α=0.1, ε=0.4 (decaying)

**Outputs:**
- Training GIFs for selected episodes
- Learning curves and optimal policy visualization
- Performance statistics

### 2. Grid Size Impact Analysis

**File:** `grid_size_impact_analysis.py`

**Purpose:** Study how grid dimensions affect learning performance

**Grid Sizes Tested:** 4x4, 6x6, 8x8, 10x10

**Metrics:**
- Cumulative rewards per episode
- Q-table convergence (quadratic difference)
- Success rate (goal achievement percentage)
- Average episode length

**Key Insights:**
- Larger grids require more episodes for convergence
- Increased complexity with grid size
- Higher exploration needs for larger environments

### 3. Hyperparameter Sensitivity Analysis

**File:** `hyperparameter_sensitivity_analysis.py`

**Purpose:** Evaluate parameter influence on learning performance

**Parameters Tested:**
- **Discount factor (γ):** 0.5, 0.7, 0.9, 0.99
- **Learning rate (α):** 0.1, 0.3, 0.5, 0.7
- **Exploration strategies (ε):** Constant (0.1), Decaying, High (0.3)

**Findings:**
- High γ: Better long-term vision but unstable learning
- High α: Faster learning but potential instability
- Decaying ε: Optimal exploration/exploitation balance

### 4. Dynamic Goal Environment

**File:** `dynamic_goal_qlearning.py`

**Purpose:** Test Q-learning adaptability in non-stationary environments

**Experimental Setup:**
- Fixed 8x8 grid with static obstacles
- Goal changes every 100 episodes
- 12 possible goal positions
- Agent must relearn policy after each change

**Key Observations:**
- Performance drops after each goal change
- Q-table instability during transitions
- Limited generalization capability
- Need for complete relearning rather than adaptation

## Environment Configurations

### Static Environments

**4x4 Grid:**
- Terminal states: (0,0), (3,3)
- Obstacles: (1,1), (2,2)

**6x6 Grid:**
- Terminal states: (0,0), (5,5)
- Obstacles: (1,1), (2,2), (4,3), (3,4)

**8x8 Grid:**
- Terminal states: (0,0), (7,7)
- Obstacles: (1,1), (2,2), (4,3), (3,4), (6,5), (5,6)

**10x10 Grid:**
- Terminal states: (0,0), (9,9)
- Obstacles: (1,1), (2,2), (4,3), (3,4), (6,5), (5,6), (8,7), (7,8)

### Dynamic Environment
- **Grid:** 8x8
- **Obstacles:** Fixed positions
- **Goal positions:** 12 possible locations
- **Change interval:** Every 100 episodes

### Reward System
- Goal achievement: +10
- Per step: -1
- Obstacle collision: -2
- Grid boundary: -2

## Performance Metrics

### Learning Performance
- **Cumulative reward:** Total reward per episode
- **Success rate:** Percentage of episodes reaching goal
- **Episode length:** Average steps per episode

### Convergence Analysis
- **Q-table convergence:** Norm of differences between successive Q-tables
- **Stability:** Performance variance across final episodes

### Efficiency Measures
- **Convergence speed:** Episodes needed for stable policy
- **Final performance:** Average metrics over last 100 episodes

## Key Findings

### Grid Size Impact
- Exponential increase in learning time with grid size
- Memory requirements grow with state space
- Exploration strategies become crucial for larger grids

### Hyperparameter Sensitivity
- Optimal γ around 0.9 for balance between short and long-term planning
- α=0.1 provides stable learning without oscillations
- Decaying ε strategy outperforms constant exploration

### Dynamic Environment Limitations
- Q-learning struggles with non-stationary environments
- Complete policy relearning required after changes
- Poor generalization across different goal positions
- Highlights need for more adaptive algorithms

## Usage Examples

### Basic Implementation
```bash
python 03_Level2_TD/q_learning.py
```

### Grid Size Study
```bash
python 03_Level2_TD/grid_size_impact_analysis.py
```

### Parameter Analysis
```bash
python 03_Level2_TD/hyperparameter_sensitivity_analysis.py
```

### Dynamic Environment Test
```bash
python 03_Level2_TD/dynamic_goal_qlearning.py
```

## Output Files

### Visualizations
- `learning_curves.png`: Reward and length progression
- `optimal_policy_*.png`: Learned policy visualization
- `grid_size_comparison.png`: Multi-grid performance comparison
- `hyperparameter_*_comparison.png`: Parameter influence analysis
- `dynamic_goal_performance.png`: Adaptation capability assessment

### Training Animations
- GIF files showing agent behavior at different training stages
- Episodes: Initial, middle, final, and transition periods

### Console Output
- Training progress updates
- Final performance statistics
- Policy extraction details
- Comparative analysis results

## Customization

### Environment Modification
Edit configuration in main functions:
- Grid dimensions and obstacle positions
- Reward structure and penalties
- Terminal state locations

### Parameter Tuning
Adjust learning parameters:
- Learning rate (α) and discount factor (γ)
- Exploration strategies and decay rates
- Episode limits and performance thresholds

### Experimental Extensions
- Add new grid sizes or obstacle configurations
- Test additional hyperparameter values
- Implement alternative exploration strategies
- Extend to more complex environment dynamics

## Technical Notes

### Algorithm Implementation
- Tabular Q-learning with discrete state-action space
- Epsilon-greedy exploration with automatic decay
- Obstacle handling through environment penalties
- Efficient state representation and Q-table updates

### Visualization Features
- High-quality matplotlib rendering
- Animated GIF creation with PIL
- Comprehensive policy representation
- Comparative analysis plots

## Limitations and Future Work

### Current Limitations
- Computational complexity with large state spaces
- Sensitivity to initial conditions
- Memory requirements for Q-table storage
- Limited adaptability to environmental changes

### Potential Improvements
- Q-learning with function approximation
- Advanced exploration strategies (UCB, Thompson sampling)
- Transfer learning for dynamic environments
- Multi-goal reinforcement learning
- Statistical significance testing
- Parallel experimentation capabilities


## Conclusion

This comprehensive study demonstrates Q-learning capabilities and limitations across various GridWorld environments. The experiments provide valuable insights into parameter tuning, scalability issues, and adaptation challenges in dynamic settings, serving as a foundation for more advanced reinforcement learning research.