import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.animation import FuncAnimation, PillowWriter
import io
from PIL import Image

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from GridWorld.grid_env import GW

def epsilon_greedy_policy(Q, state, epsilon, n_actions):
    """Epsilon-greedy policy"""
    x, y = state
    if np.random.random() > epsilon:
        return np.argmax(Q[x, y, :])
    else:
        return np.random.randint(n_actions)

def run_q_learning_experiment(grid_size, num_episodes=1000, alpha=0.1, gamma=0.9, epsilon=0.2):
    """Run Q-learning for a given grid size"""
    
    # Environment configuration depending on the size
    if grid_size == 4:
        obstacles = [(1, 1), (2, 2)]
        terminal_states = [(0, 0), (3, 3)]
    elif grid_size == 6:
        obstacles = [(1, 1), (2, 2), (4, 3), (3, 4)]
        terminal_states = [(0, 0), (5, 5)]
    elif grid_size == 8:
        obstacles = [(1, 1), (2, 2), (4, 3), (3, 4), (6, 5), (5, 6)]
        terminal_states = [(0, 0), (7, 7)]
    else:  # 10x10
        obstacles = [(1, 1), (2, 2), (4, 3), (3, 4), (6, 5), (5, 6), (8, 7), (7, 8)]
        terminal_states = [(0, 0), (9, 9)]
    
    env = GW(
        grid_size=grid_size,
        terminal_states=terminal_states,
        obstacle_positions=obstacles,
        step_penalty=-1,
        goal_reward=10,
        obstacle_penalty=-2,
        max_steps=grid_size * 10,
        show_agent=False
    )
    
    n_actions = env.action_space.n
    Q = np.zeros((grid_size, grid_size, n_actions))
    
    episode_rewards = []
    episode_lengths = []
    q_convergence = []  # Evolution of the Q-table
    success_rate = []   # Success rate
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        reached_goal = False
        
        while not (terminated or truncated):
            action = epsilon_greedy_policy(Q, state, epsilon, n_actions)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            x, y = state
            nx, ny = next_state
            
            target = reward + gamma * np.max(Q[nx, ny, :])
            Q[x, y, action] += alpha * (target - Q[x, y, action])
            
            state = next_state
            total_reward += reward
            steps += 1
            
            if terminated and reward == 10:  # Goal atteint
                reached_goal = True
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_rate.append(1 if reached_goal else 0)
        
        # Convergence measure of the Q-table (difference between episodes)
        if episode > 0:
            q_diff = np.sqrt(np.sum((Q - previous_Q) ** 2))
            q_convergence.append(q_diff)
        previous_Q = Q.copy()
        
        # Decaying epsilon
        current_epsilon = max(0.01, epsilon * (1 - episode/num_episodes))
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'q_convergence': q_convergence,
        'success_rate': success_rate,
        'final_Q': Q
    }

def plot_grid_size_comparison(results):
    """Plot comparisons for different grid sizes"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange']
    grid_sizes = [4, 6, 8, 10]
    
    # 1. Cumulative rewards curve
    for i, size in enumerate(grid_sizes):
        rewards = results[size]['episode_rewards']
        # Moving average for smoothing
        window = 50
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 
                color=colors[i], linewidth=2, label=f'{size}x{size}')
    
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense cumulée (moyenne mobile)')
    ax1.set_title('Récompenses cumulées par taille de grille')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-table convergence
    for i, size in enumerate(grid_sizes):
        q_conv = results[size]['q_convergence']
        ax2.plot(q_conv, color=colors[i], linewidth=2, label=f'{size}x{size}')
    
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Q-table convergence gap')
    ax2.set_title('Q-table convergence')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')  # Log scale to better see differences
    
    # 3. Success rate (moving average)
    for i, size in enumerate(grid_sizes):
        success = results[size]['success_rate']
        window = 100
        success_avg = np.convolve(success, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(success)), success_avg, 
                color=colors[i], linewidth=2, label=f'{size}x{size}')
    
    ax3.set_xlabel('Épisode')
    ax3.set_ylabel('Taux de succès')
    ax3.set_title('Taux de succès (moyenne mobile sur 100 épisodes)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Average episode length
    for i, size in enumerate(grid_sizes):
        lengths = results[size]['episode_lengths']
        window = 50
        length_avg = np.convolve(lengths, np.ones(window)/window, mode='valid')
        ax4.plot(range(window-1, len(lengths)), length_avg, 
                color=colors[i], linewidth=2, label=f'{size}x{size}')
    
    ax4.set_xlabel('Épisode')
    ax4.set_ylabel('Longueur moyenne des épisodes')
    ax4.set_title('Longueur des épisodes par taille de grille')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('03_Level2_TD/q_learning_results/grid_size_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistiques finales
    print("STATISTIQUES FINALES PAR TAILLE DE GRILLE")
    print("=" * 50)
    for size in grid_sizes:
        data = results[size]
        final_success = np.mean(data['success_rate'][-100:]) * 100
        final_reward = np.mean(data['episode_rewards'][-100:])
        print(f"Grille {size}x{size}:")
        print(f"  - Taux de succès final: {final_success:.1f}%")
        print(f"  - Récompense moyenne finale: {final_reward:.2f}")
        print(f"  - Convergence Q-table finale: {data['q_convergence'][-1]:.6f}")
        print()

def main():
    """Expérience : Influence de la taille de grille"""
    print("Début de l'expérience : Influence de la taille de grille")
    print("=" * 60)
    
    grid_sizes = [4, 6, 8, 10]
    num_episodes = 1500
    results = {}
    
    for grid_size in grid_sizes:
        print(f"Traitement de la grille {grid_size}x{grid_size}...")
        results[grid_size] = run_q_learning_experiment(
            grid_size=grid_size,
            num_episodes=num_episodes,
            alpha=0.1,
            gamma=0.9,
            epsilon=0.2
        )
    
    # Generate plots
    plot_grid_size_comparison(results)
    
    print("Expérience terminée. Graphiques sauvegardés.")

if __name__ == "__main__":
    main()