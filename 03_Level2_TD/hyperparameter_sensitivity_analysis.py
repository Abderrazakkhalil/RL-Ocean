import numpy as np
import matplotlib.pyplot as plt
import os
import sys

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

def run_hyperparameter_experiment(alpha, gamma, epsilon_strategy, grid_size=6, num_episodes=1000):
    """Run Q-learning with specific hyperparameters"""
    
    # Fixed environment configuration
    obstacles = [(1, 1), (2, 2), (4, 3), (3, 4)]
    terminal_states = [(0, 0), (5, 5)]
    
    env = GW(
        grid_size=grid_size,
        terminal_states=terminal_states,
        obstacle_positions=obstacles,
        step_penalty=-1,
        goal_reward=10,
        obstacle_penalty=-2,
        max_steps=100,
        show_agent=False
    )
    
    n_actions = env.action_space.n
    Q = np.zeros((grid_size, grid_size, n_actions))
    
    episode_rewards = []
    episode_lengths = []
    q_convergence = []
    success_rate = []
    
    for episode in range(num_episodes):
        # Handle epsilon strategy
        if epsilon_strategy == 'constant':
            epsilon = 0.1
        elif epsilon_strategy == 'decaying':
            epsilon = max(0.01, 0.3 * (1 - episode/num_episodes))
        else:  # high_epsilon
            epsilon = 0.3
        
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
            
            if terminated and reward == 10:
                reached_goal = True
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_rate.append(1 if reached_goal else 0)
        
        if episode > 0:
            q_diff = np.sqrt(np.sum((Q - previous_Q) ** 2))
            q_convergence.append(q_diff)
        previous_Q = Q.copy()
    
    env.close()
    
    return {
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'q_convergence': q_convergence,
        'success_rate': success_rate,
        'final_Q': Q
    }

def plot_hyperparameter_comparison(results, param_name, param_values):
    """Plot comparisons for different hyperparameters"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_values)))
    
    # 1. Cumulative rewards (last 100 episodes)
    final_rewards = []
    for i, param_value in enumerate(param_values):
        rewards = results[param_value]['episode_rewards']
        # Moyenne des 100 derniers épisodes
        final_avg = np.mean(rewards[-100:])
        final_rewards.append(final_avg)
        
        # Full curve (moving average)
        window = 50
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 
                color=colors[i], linewidth=2, label=f'{param_name}={param_value}')
    
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense cumulée')
    ax1.set_title(f'Récompenses cumulées - Variation de {param_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-table convergence
    for i, param_value in enumerate(param_values):
        q_conv = results[param_value]['q_convergence']
        ax2.plot(q_conv, color=colors[i], linewidth=2, label=f'{param_name}={param_value}')
    
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Q-table convergence gap')
    ax2.set_title(f'Q-table convergence - Variation of {param_name}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Final success rate
    final_success_rates = []
    for param_value in param_values:
        success = results[param_value]['success_rate']
        final_success = np.mean(success[-100:]) * 100
        final_success_rates.append(final_success)
    
    bars = ax3.bar(range(len(param_values)), final_success_rates, color=colors, alpha=0.7)
    ax3.set_xlabel(param_name)
    ax3.set_ylabel('Taux de succès final (%)')
    ax3.set_title(f'Taux de succès final - Variation de {param_name}')
    ax3.set_xticks(range(len(param_values)))
    ax3.set_xticklabels([str(v) for v in param_values])
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars, final_success_rates):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 4. Final average reward
    bars = ax4.bar(range(len(param_values)), final_rewards, color=colors, alpha=0.7)
    ax4.set_xlabel(param_name)
    ax4.set_ylabel('Récompense moyenne finale')
    ax4.set_title(f'Récompense moyenne finale - Variation de {param_name}')
    ax4.set_xticks(range(len(param_values)))
    ax4.set_xticklabels([str(v) for v in param_values])
    
    # Ajouter les valeurs sur les barres
    for bar, value in zip(bars, final_rewards):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'03_Level2_TD/q_learning_results/hyperparameter_{param_name}_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def experiment_gamma():
    """Experiment: Vary gamma"""
    print("Expérience : Variation du facteur de discount gamma")
    print("=" * 50)
    
    gamma_values = [0.5, 0.7, 0.9, 0.99]
    results = {}
    
    for gamma in gamma_values:
        print(f"Test avec gamma = {gamma}")
        results[gamma] = run_hyperparameter_experiment(
            alpha=0.1,
            gamma=gamma,
            epsilon_strategy='decaying'
        )
    
    plot_hyperparameter_comparison(results, 'gamma', gamma_values)
    
    # Analysis
    print("\nANALYSE GAMMA:")
    for gamma in gamma_values:
        data = results[gamma]
        final_success = np.mean(data['success_rate'][-100:]) * 100
        final_reward = np.mean(data['episode_rewards'][-100:])
        print(f"Gamma {gamma}: Succès = {final_success:.1f}%, Récompense = {final_reward:.2f}")

def experiment_alpha():
    """Experiment: Vary alpha"""
    print("\nExpérience : Variation du taux d'apprentissage alpha")
    print("=" * 50)
    
    alpha_values = [0.1, 0.3, 0.5, 0.7]
    results = {}
    
    for alpha in alpha_values:
        print(f"Test avec alpha = {alpha}")
        results[alpha] = run_hyperparameter_experiment(
            alpha=alpha,
            gamma=0.9,
            epsilon_strategy='decaying'
        )
    
    plot_hyperparameter_comparison(results, 'alpha', alpha_values)
    
    # Analysis
    print("\nANALYSE ALPHA:")
    for alpha in alpha_values:
        data = results[alpha]
        final_success = np.mean(data['success_rate'][-100:]) * 100
        final_reward = np.mean(data['episode_rewards'][-100:])
        convergence = data['q_convergence'][-1] if len(data['q_convergence']) > 0 else 0
        print(f"Alpha {alpha}: Succès = {final_success:.1f}%, Récompense = {final_reward:.2f}, Convergence = {convergence:.6f}")

def experiment_epsilon():
    """Experiment: Exploration strategies"""
    print("\nExpérience : Stratégies d'exploration epsilon")
    print("=" * 50)
    
    epsilon_strategies = ['constant', 'decaying', 'high_epsilon']
    strategy_names = ['Constant (0.1)', 'Décroissant', 'Haut (0.3)']
    results = {}
    
    for strategy in epsilon_strategies:
        print(f"Test avec stratégie: {strategy}")
        results[strategy] = run_hyperparameter_experiment(
            alpha=0.1,
            gamma=0.9,
            epsilon_strategy=strategy
        )
    
    # Adaptation pour le plotting
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    colors = ['blue', 'red', 'green']
    
    # 1. Récompenses cumulées
    for i, strategy in enumerate(epsilon_strategies):
        rewards = results[strategy]['episode_rewards']
        window = 50
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(rewards)), moving_avg, 
                color=colors[i], linewidth=2, label=strategy_names[i])
    
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense cumulée')
    ax1.set_title('Récompenses cumulées - Stratégies epsilon')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Taux de succès (moyenne mobile)
    for i, strategy in enumerate(epsilon_strategies):
        success = results[strategy]['success_rate']
        window = 100
        success_avg = np.convolve(success, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(success)), success_avg, 
                color=colors[i], linewidth=2, label=strategy_names[i])
    
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Taux de succès')
    ax2.set_title('Taux de succès - Stratégies epsilon')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Taux de succès final
    final_success = [np.mean(results[s]['success_rate'][-100:]) * 100 for s in epsilon_strategies]
    bars = ax3.bar(strategy_names, final_success, color=colors, alpha=0.7)
    ax3.set_ylabel('Taux de succès final (%)')
    ax3.set_title('Taux de succès final - Stratégies epsilon')
    
    for bar, value in zip(bars, final_success):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # 4. Récompense moyenne finale
    final_rewards = [np.mean(results[s]['episode_rewards'][-100:]) for s in epsilon_strategies]
    bars = ax4.bar(strategy_names, final_rewards, color=colors, alpha=0.7)
    ax4.set_ylabel('Récompense moyenne finale')
    ax4.set_title('Récompense moyenne finale - Stratégies epsilon')
    
    for bar, value in zip(bars, final_rewards):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('03_Level2_TD/q_learning_results/hyperparameter_epsilon_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Analyse
    print("\nANALYSE EPSILON:")
    for i, strategy in enumerate(epsilon_strategies):
        data = results[strategy]
        final_success = np.mean(data['success_rate'][-100:]) * 100
        final_reward = np.mean(data['episode_rewards'][-100:])
        print(f"Stratégie {strategy_names[i]}: Succès = {final_success:.1f}%, Récompense = {final_reward:.2f}")

def main():
    """Expériences complètes sur les hyperparamètres"""
    print("Début des expériences sur les hyperparamètres")
    print("=" * 60)
    
    experiment_gamma()
    experiment_alpha()
    experiment_epsilon()
    
    print("\nToutes les expériences sont terminées.")

if __name__ == "__main__":
    main()