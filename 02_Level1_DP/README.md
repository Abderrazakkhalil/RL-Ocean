# Dynamic Programming - Policy Iteration Analysis

## Project Overview

This project implements and analyzes **Policy Iteration** - a fundamental Dynamic Programming algorithm for solving Markov Decision Processes (MDPs). The implementation includes performance benchmarking, parameter analysis, and comprehensive visualization for grid world environments.

## Project Structure

```
02_Level1_DP/
├── PI_env.py                 # Custom GridWorld environment for Policy Iteration
├── policy_iteration.py       # Main Policy Iteration algorithm implementation
├── performance_comparison.py # Performance analysis across grid sizes
├── parameter_comparison.py   # Parameter sensitivity analysis
└── results/                  # Generated results and visualizations
```

## Core Components

### 1. PI_env.py - Custom Environment

**Specialized GridWorld environment for Policy Iteration analysis:**

- **Grid Size**: 4x4 by default (configurable)
- **Terminal States**: (0,0) and (3,3)
- **Actions**: 4-directional movement (Up, Down, Right, Left)
- **Rewards**:
  - Goal state: +1
  - Movement cost: -1
  - Wall collision: -2

**Key Methods:**

- `step_state(state, action)`: Deterministic state transitions
- `render(V, policy, iteration)`: Custom visualization with value function display

### 2. policy_iteration.py - Main Algorithm

**Implements recursive Policy Iteration with detailed logging:**

#### Algorithm Steps:

1. **Policy Evaluation**: Iteratively compute value function for current policy
2. **Policy Improvement**: Extract improved policy using value function
3. **Convergence Check**: Stop when policy becomes stable

#### Key Functions:

- `policy_iteration()`: Main recursive implementation
- `policy_evaluation()`: Bellman equation solver
- `extract_policy()`: Greedy policy improvement
- `extract_all_optimal_actions()`: Finds all optimal actions for visualization

#### Features:

- **Recursive implementation** with depth control
- **Real-time visualization** of value function evolution
- **Comprehensive logging** of each iteration
- **Multiple optimal actions** detection
- **Automatic result saving**

### 3. performance_comparison.py - Scalability Analysis

**Benchmarks Policy Iteration across different grid sizes:**

#### Tested Grid Sizes:

- 4x4, 8x8, 16x16, 32x32

#### Performance Metrics:

- **Execution time** (total and per iteration)
- **Iteration count** to convergence
- **Memory usage** patterns
- **Computational complexity** analysis

#### Output:

- Performance comparison charts
- Convergence analysis plots
- Detailed performance report (CSV)

### 4. parameter_comparison.py - Hyperparameter Analysis

**Comprehensive analysis of algorithm parameters:**

#### Parameters Tested:

- **Gamma (γ)**: [0.5, 0.6, 0.7, 0.9, 0.95, 0.99, 0.999] - Discount factor
- **Theta (θ)**: [1e-1, 1e-2, 1e-3, 1e-4, 1e-5] - Convergence threshold
- **Evaluation Methods**: Synchronous vs Asynchronous updates

#### Analysis Includes:

- **Convergence speed** vs parameter values
- **Solution quality** impact
- **Computational efficiency** trade-offs
- **Optimal parameter** recommendations

## Usage Examples

### Basic Policy Iteration

```python
from PI_env import PI
from policy_iteration import policy_iteration

env = PI()
env.reset()

# Run Policy Iteration
V, policy = policy_iteration(env, gamma=0.9)

# Display results
env.render(V=V, policy=policy)
```

### Performance Benchmarking

```python
from performance_comparison import run_performance_comparison

# Compare across grid sizes
results = run_performance_comparison(gamma=0.9, theta=1e-3)
```

### Parameter Analysis

```python
from parameter_comparison import run_comprehensive_comparison

# Analyze parameter sensitivity
parameter_results = run_comprehensive_comparison()
```

## Key Features

### Visualization Capabilities

- **Value Function Heatmaps**: Color-coded state values
- **Policy Arrows**: Direction indicators for optimal actions
- **Iteration Progression**: Step-by-step algorithm visualization
- **Convergence Plots**: Policy changes over iterations

### Analysis Tools

- **Automatic Report Generation**: CSV and visual reports
- **Complexity Analysis**: O(n⁴) computational complexity verification
- **Convergence Detection**: Automatic stopping criteria
- **Multiple Solutions**: Handling of ties in optimal actions

### Algorithmic Enhancements

- **Recursive Implementation**: Clean, educational code structure
- **Early Termination**: Stops when policy stabilizes
- **Flexible Parameters**: Easy experimentation with different settings
- **Detailed Logging**: Step-by-step algorithm tracing

## Results and Insights

### Performance Characteristics

- **Time Complexity**: O(n⁴) observed for Policy Iteration
- **Space Complexity**: O(n²) for value function storage
- **Convergence**: Typically 3-5 iterations for 4x4 grid
- **Scalability**: Performance degrades rapidly with grid size

### Parameter Recommendations

- **Optimal Gamma**: 0.9 (balances future reward consideration)
- **Recommended Theta**: 1e-3 (good precision/efficiency trade-off)
- **Best Method**: Synchronous evaluation (more stable)

### Algorithm Behavior

- **Policy Stability**: Converges to optimal policy in finite steps
- **Value Convergence**: Monotonic improvement guaranteed
- **Solution Quality**: Finds truly optimal policies
- **Grid Size Impact**: Computation time grows polynomially

## Technical Details

### Dependencies

- Python 3.7+
- NumPy
- Matplotlib
- Pandas
- Gymnasium

### Installation

```bash
pip install numpy matplotlib pandas gymnasium
```

### Output Files

- `results/performance_comparison.png` - Grid size scalability
- `results/convergence_analysis.png` - Convergence patterns
- `results/parameter_comparison_report.csv` - Detailed parameter analysis
- `results/optimal_policy_all_actions.png` - Final policy visualization

## Educational Value

This implementation serves as an excellent educational resource for:

- **Understanding Dynamic Programming** fundamentals
- **Policy Iteration algorithm** mechanics
- **MDP solution methods** comparison
- **Algorithm complexity** analysis
- **Reinforcement Learning** foundations

## Research Applications

The codebase provides a foundation for:

- **Algorithm benchmarking** against other MDP solvers
- **Parameter sensitivity** studies
- **Large-scale MDP** solution techniquesw
- **Educational demonstrations** of DP concepts
- **Custom environment** development and testing

---

_Note: This project focuses specifically on Policy Iteration analysis. The GridWorld environment (`GridWorld/grid_env.py`) is used as a dependency but not directly modified or analyzed in this specific implementation._
