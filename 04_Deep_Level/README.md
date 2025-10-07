# Deep Q-Learning for GridWorld

## Project Overview

This project implements Deep Q-Learning (DQN), a deep reinforcement learning algorithm, for solving navigation tasks in a GridWorld environment. The algorithm uses a neural network to approximate Q-values instead of storing them in a table, enabling handling of larger state spaces and potential generalization.

## Project Structure

```
RL_ocean/
├── 03_Level2_TD/
│   ├── deep_q_learning.py                    # Main Deep Q-Learning implementation
│   ├── q_learning.py                         # Traditional Q-learning
│   ├── grid_size_impact_analysis.py          # Grid size study
│   ├── hyperparameter_sensitivity_analysis.py # Parameter tuning
│   ├── dynamic_goal_qlearning.py             # Dynamic environment study
│   └── results/
│       ├── deep_q_learning_results/
│       │   ├── training_episode_*.gif
│       │   ├── deep_learning_curves.png
│       │   └── neural_network_analysis.png
│       ├── q_learning_results/
│       └── ... (other result folders)
└── GridWorld/
    └── grid_env.py
```

## Core Features

### Deep Q-Learning Algorithm
- Neural network-based Q-value approximation
- Experience replay for stable training
- Target network to reduce correlation
- Epsilon-greedy exploration with decay
- Minibatch training from replay memory

### Neural Network Architecture
- **Input Layer**: One-hot encoded state (36 neurons for 6x6 grid)
- **Hidden Layers**: 2 fully connected layers with ReLU activation
- **Output Layer**: 4 neurons (Q-values for each action)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam

### Environment Configuration
- **Grid Size**: 6x6
- **Terminal States**: (0,0) and (5,5)
- **Obstacles**: (1,1), (2,2), (4,3), (3,4)
- **Rewards**: Goal (+10), Step penalty (-1), Obstacle penalty (-2)
- **Max Steps**: 100 per episode

## Dependencies

- Python 3.7+
- NumPy
- Matplotlib
- Pillow (PIL)
- PyTorch
- GridWorld environment

## Installation

```bash
pip install numpy matplotlib pillow torch
```

## Usage

### Basic Execution

```bash
python 03_Level2_TD/deep_q_learning.py
```

### Key Parameters

- **Number of episodes**: 800
- **Learning rate (α)**: 0.001
- **Discount factor (γ)**: 0.99
- **Initial exploration (ε)**: 0.3
- **Epsilon decay**: 0.995
- **Minimum epsilon**: 0.01
- **Hidden layer size**: 64 neurons
- **Batch size**: 32
- **Replay memory**: 2000 experiences

## Algorithm Components

### State Representation
```python
def state_to_features(state, grid_size=6):
    # One-hot encoding of position
    features = np.zeros(grid_size * grid_size)
    position_index = x * grid_size + y
    features[position_index] = 1.0
    return features
```

### Neural Network
```python
class DeepQNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
```

### Training Loop
1. **Experience Collection**: Store (state, action, reward, next_state, done) in replay memory
2. **Minibatch Sampling**: Randomly sample experiences from memory
3. **Q-value Prediction**: Use Q-network for current Q-values
4. **Target Calculation**: Use target network for next Q-values
5. **Loss Optimization**: Update Q-network weights via backpropagation
6. **Target Update**: Periodically sync target network with Q-network

## Output Files

### Generated Visualizations

1. **Training GIFs**:
   - `training_episode_1.gif`: Initial exploration behavior
   - `training_episode_400.gif`: Mid-training performance
   - `training_episode_800.gif`: Final learned policy

2. **Performance Plots**:
   - `deep_learning_curves.png`: Rewards, lengths, losses, and success rates
   - `neural_network_analysis.png`: Q-value distributions and correlations

3. **Console Output**:
   - Training progress every 100 episodes
   - Neural network predictions analysis
   - Final performance statistics

### Key Metrics Tracked

- **Episode Rewards**: Cumulative reward per episode
- **Episode Lengths**: Number of steps per episode
- **Training Loss**: Neural network optimization progress
- **Success Rate**: Percentage of episodes reaching goal
- **Memory Usage**: Size of experience replay buffer
- **Exploration Rate**: Current epsilon value

## Deep Q-Learning Mechanisms

### Experience Replay
- Breaks temporal correlations between experiences
- Enables reuse of past experiences
- Improves sample efficiency
- Provides more stable training

### Target Network
- Reduces correlation between Q-values and targets
- Provides stable learning targets
- Updated periodically from Q-network
- Prevents oscillation and divergence

### Epsilon-Greedy Strategy
- Balances exploration and exploitation
- Starts with high exploration (ε=0.3)
- Decays to minimum exploration (ε=0.01)
- Ensures adequate state space coverage

## Comparison with Traditional Q-Learning

### Advantages of Deep Q-Learning
- **Memory Efficiency**: No need for large Q-table (O(n²) vs O(1) in network parameters)
- **Generalization**: Can handle unseen states through function approximation
- **Scalability**: Suitable for larger state spaces
- **Feature Learning**: Automatically learns relevant state representations

### Challenges of Deep Q-Learning
- **Training Stability**: More sensitive to hyperparameters
- **Sample Efficiency**: Requires more experiences
- **Computational Cost**: Neural network forward/backward passes
- **Convergence Guarantees**: Less theoretical guarantees than tabular Q-learning

## Performance Analysis

### Expected Learning Curve
- **Initial Phase**: Random exploration, negative rewards
- **Learning Phase**: Gradual improvement in rewards and success rate
- **Convergence Phase**: Stable policy with high success rate
- **Loss Behavior**: Decreasing loss with occasional spikes during exploration

### Neural Network Insights
- **Q-value Distribution**: Shows confidence in different actions
- **Action Correlations**: Reveals relationships between action values
- **Prediction Analysis**: Demonstrates generalization capability
- **Network Stability**: Indicated by smooth loss decrease

## Customization

### Modifying Network Architecture
```python
# Change hidden layer size
agent = DeepQLearningAgent(state_size, action_size, hidden_size=128)

# Modify network layers in DeepQNetwork class
self.fc1 = nn.Linear(input_size, 256)
self.fc2 = nn.Linear(256, 128)
self.fc3 = nn.Linear(128, output_size)
```

### Adjusting Training Parameters
```python
results = deep_q_learning(
    env=env,
    num_episodes=1000,
    gamma=0.95,           # Change discount factor
    alpha=0.0005,         # Change learning rate
    epsilon=0.2,          # Change initial exploration
    batch_size=64         # Change batch size
)
```

### Environment Modifications
- Change grid size and update state representation
- Add/remove obstacles and terminal states
- Modify reward structure
- Adjust maximum episode length

## Technical Considerations

### Hardware Requirements
- **CPU**: Sufficient for small networks and grids
- **GPU**: Beneficial for larger networks or more complex environments
- **Memory**: Replay buffer size affects RAM usage

### Training Stability Tips
- Use smaller learning rates for deep networks
- Ensure adequate replay buffer size
- Regular target network updates
- Monitor loss and reward curves for signs of divergence

### Common Issues and Solutions
- **NaN Loss**: Reduce learning rate, check reward scaling
- **No Learning**: Increase exploration, check reward function
- **Oscillating Performance**: Increase batch size, decrease learning rate
- **Overfitting**: Add regularization, increase replay buffer size

## Extensions and Future Work

### Algorithm Improvements
- Double DQN for more stable value estimation
- Dueling DQN for separate value and advantage streams
- Prioritized experience replay for important samples
- Noisy networks for better exploration

### Application Areas
- Larger grid worlds with more complex obstacles
- Multi-agent environments
- Partial observability (POMDPs)
- Continuous action spaces with actor-critic methods


## Conclusion

This Deep Q-Learning implementation demonstrates the power of neural networks in reinforcement learning, showing how function approximation can overcome the limitations of tabular methods while introducing new challenges in training stability and hyperparameter tuning.