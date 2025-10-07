import numpy as np 
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.animation import FuncAnimation, PillowWriter
import io
from PIL import Image

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the environment from the renamed package
from GridWorld.grid_env import GW

def epsilon_greedy_policy(Q, state, epsilon, n_actions, obstacles):
    """
    Epsilon-greedy policy based on Q-values
    
    Args:
        Q: Q-table (n x n x n_actions)
        state: Current state (x, y)
        epsilon: Exploration probability
        n_actions: Number of possible actions
        obstacles: List of obstacle positions (informational only)
    
    Returns:
        action: Chosen action
    """
    x, y = state
    
    # DO NOT prevent choosing actions towards obstacles
    # The GW environment already handles obstacles with obstacle_penalty
    
    # With probability 1-epsilon, choose the optimal action (exploitation)
    if np.random.random() > epsilon:
        return np.argmax(Q[x, y, :])
    # With probability epsilon, choose a random action (exploration)
    else:
        return np.random.randint(n_actions)

def save_frame(fig):
    """
    Save a frame from a matplotlib figure with better quality
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

def q_learning(env, num_episodes=1000, gamma=0.9, alpha=0.1, epsilon=0.1, 
               epsilon_decay=0.995, min_epsilon=0.01, save_gif_episodes=None):
    """
    Implementation of the Q-learning algorithm for GridWorld
    RESPECTING the obstacle behavior defined in GW
    """
    
    # Initialize the Q-table
    n = env.grid_width
    m = env.grid_height
    n_actions = env.action_space.n
    Q = np.zeros((n, m, n_actions))
    
    # DO NOT initialize obstacle Q-values to -inf
    # Obstacles are normal states with negative rewards
    
    # Track performance
    episode_rewards = []
    episode_lengths = []
    
    # Episodes for which to save GIFs
    if save_gif_episodes is None:
        save_gif_episodes = [0, num_episodes//2, num_episodes-1]
    
    gif_data = {}
    
    print("Début de l'apprentissage Q-learning...")
    print(f"Grille: {n}x{m}, Actions: {n_actions}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Terminal states: {env.terminal_states}")
    print(f"Paramètres: gamma={gamma}, alpha={alpha}, epsilon={epsilon}")
    print("=" * 50)
    
    for episode in range(num_episodes):
        # Environment reset
        state, info = env.reset()
        
        # PROTECTION: Ensure the initial state is NOT an obstacle
        max_retries = 10
        retry_count = 0
        while tuple(state) in env.obstacles and retry_count < max_retries:
            print(f"⚠️  Agent initialisé sur obstacle {tuple(state)}, réessai {retry_count+1}/{max_retries}")
            state, info = env.reset()
            retry_count += 1
        
        if retry_count >= max_retries:
            print(f"❌ ERREUR: Impossible d'initialiser l'agent hors des obstacles")
            # Force a safe position
            safe_positions = [(x, y) for x in range(n) for y in range(m) 
                            if (x, y) not in env.obstacles and (x, y) not in env.terminal_states]
            if safe_positions:
                state = np.array(safe_positions[0])
                print(f"✅ Agent forcé à la position safe: {state}")
            else:
                print("❌ Aucune position safe disponible!")
                continue
        
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        # Store frames for the GIF
        if episode in save_gif_episodes:
            frames = []
            # Save the initial frame
            env.render()
            plt.pause(0.01)
            frames.append(save_frame(plt.gcf()))
        
        # Episode loop
        while not (terminated or truncated):
            # Choose action according to epsilon-greedy policy
            action = epsilon_greedy_policy(Q, state, epsilon, n_actions, env.obstacles)
            
            # Execute the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Q-learning update (off-policy)
            x, y = state
            nx, ny = next_state
            
            # Standard Q-learning target: r + gamma * max_a' Q(s', a')
            # DO NOT interfere with obstacle behavior
            # The GW environment already handles rewards and transitions
            target = reward + gamma * np.max(Q[nx, ny, :])
            
            # Update the Q-value
            Q[x, y, action] += alpha * (target - Q[x, y, action])
            
            # Update for the next iteration
            state = next_state
            total_reward += reward
            steps += 1
            
            # Save frame for GIF (every step for selected episodes)
            if episode in save_gif_episodes:
                env.render()
                plt.pause(0.01)
                frames.append(save_frame(plt.gcf()))
        
        # Save the final frame with pause
        if episode in save_gif_episodes:
            env.render()
            plt.pause(0.01)
            frames.append(save_frame(plt.gcf()))
            # Add a few duplicated final frames to pause at the end
            for _ in range(3):
                frames.append(frames[-1])
        
        # Epsilon decay
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        # Save performance data
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Save frames for GIF
        if episode in save_gif_episodes:
            gif_data[episode] = frames
            print(f"📹 {len(frames)} frames capturées pour l'épisode {episode+1}")
        
        # Progress display
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Épisode {episode+1}/{num_episodes} | "
                  f"Récompense moyenne: {avg_reward:.2f} | "
                  f"Longueur moyenne: {avg_length:.1f} | "
                  f"Epsilon: {epsilon:.3f}")
    
    # Save GIFs
    save_training_gifs(gif_data, save_gif_episodes)
    
    env.close()
    return Q, episode_rewards, episode_lengths

def save_training_gifs(gif_data, episodes):
    """
    Save training GIFs with adaptive speed
    """
    os.makedirs("03_Level2_TD/q_learning_results", exist_ok=True)
    
    for episode, frames in gif_data.items():
        if frames and len(frames) > 0:
            filename = f"03_Level2_TD/q_learning_results/training_episode_{episode+1}.gif"
            
            # Convert to PIL images
            frames_pil = [Image.fromarray(frame) for frame in frames]
            
            # Compute adaptive duration
            if len(frames_pil) < 20:
                duration = 400  # Lent pour épisodes courts
            elif len(frames_pil) < 50:
                duration = 250  # Moyen
            else:
                duration = 150  # Rapide pour épisodes longs
            
            # Save the GIF
            frames_pil[0].save(
                filename,
                save_all=True,
                append_images=frames_pil[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
            
            print(f"🎬 GIF sauvegardé: {filename} ({len(frames_pil)} frames, {duration}ms/frame)")

def extract_optimal_policy(env, Q):
    """
    Extract the optimal policy from the Q-table
    """
    n, m = env.grid_width, env.grid_height
    policy = np.zeros((n, m), dtype=int)
    policy_all = np.empty((n, m), dtype=object)
    
    action_names = ["↑", "↓", "→", "←"]
    
    print("EXTRACTION DE LA POLITIQUE OPTIMALE")
    print("=" * 40)
    
    for x in range(n):
        for y in range(m):
            # Terminal states
            if (x, y) in env.terminal_states:
                policy[x, y] = -1
                policy_all[x, y] = [-1]
                continue
            
            # Obstacles - no policy (the agent cannot be there)
            if (x, y) in env.obstacles:
                policy[x, y] = -2
                policy_all[x, y] = [-2]
                continue
            
            # Find the optimal action
            q_values = Q[x, y, :]
            max_q = np.max(q_values)
            
            # Optimal actions (those with maximal Q-value)
            best_actions = [a for a, q in enumerate(q_values) if abs(q - max_q) < 1e-6]
            
            # Deterministic policy (first optimal action)
            policy[x, y] = best_actions[0]
            
            # All optimal actions
            policy_all[x, y] = best_actions
            
            print(f"  État ({x},{y}): {[action_names[a] for a in best_actions]} "
                  f"(Q-values: {[f'{q:.3f}' for q in q_values]})")
    
    return policy, policy_all

def compute_value_function(env, Q):
    """
    Compute the value function V from the Q-table
    """
    V = np.max(Q, axis=2)
    
    # Mark obstacles with NaN for display (visual only)
    for obstacle in env.obstacles:
        x, y = obstacle
        V[x, y] = np.nan
    
    return V

def plot_learning_curve(episode_rewards, episode_lengths):
    """
    Plot learning curves
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Rewards curve
    ax1.plot(episode_rewards, alpha=0.7, linewidth=0.8, color='blue')
    
    # Moving average to smooth the curve
    window = min(100, len(episode_rewards) // 10)
    if window > 0:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                linewidth=2, color='red', label=f'Moyenne mobile ({window} épisodes)')
    
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense totale')
    ax1.set_title('Courbe d\'apprentissage - Récompenses par épisode')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Episode lengths curve
    ax2.plot(episode_lengths, alpha=0.7, linewidth=0.8, color='green')
    
    if window > 0:
        moving_avg_lengths = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), moving_avg_lengths, 
                linewidth=2, color='orange', label=f'Moyenne mobile ({window} épisodes)')
    
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Longueur de l\'épisode')
    ax2.set_title('Longueur des épisodes')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save
    os.makedirs("03_Level2_TD/q_learning_results", exist_ok=True)
    plt.savefig("03_Level2_TD/q_learning_results/learning_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_optimal_policy(env, policy_all, V, Q):
    """
    Display and save the optimal policy with ALL optimal actions
    """
    print("AFFICHAGE DE LA POLITIQUE OPTIMALE (TOUTES LES ACTIONS OPTIMALES)")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.patch.set_facecolor('#F8F9FA')
    
    n, m = env.grid_width, env.grid_height
    
    arrows = {0: '↑', 1: '↓', 2: '→', 3: '←', -1: '★', -2: '█'}
    arrow_colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 3: '#96CEB4', 
                   -1: '#FFEAA7', -2: '#666666'}
    action_names = {0: 'HAUT', 1: 'BAS', 2: 'DROITE', 3: 'GAUCHE', 
                   -1: 'TERMINAL', -2: 'OBSTACLE'}
    
    for x in range(n):
        for y in range(m):
            y_display = m - 1 - y
            
    # Cell color
            if (x, y) in env.terminal_states:
                color = "#92BD98"
                border_color = "#1F5024"
                border_width = 3
            elif (x, y) in env.obstacles:
                color = "#CCCCCC"
                border_color = "#666666"
                border_width = 2
            else:
                color = "#E3F2FD"
                border_color = "#1C4B6F"
                border_width = 2
            
            rect = plt.Rectangle((x - 0.4, y_display - 0.4), 0.8, 0.8,
                               facecolor=color, edgecolor=border_color, 
                               linewidth=border_width, alpha=0.9)
            ax.add_patch(rect)
            
            # Display all optimal actions
            if policy_all[x, y] == [-1]:
                ax.text(x, y_display, arrows[-1], ha='center', va='center', 
                       fontsize=24, fontweight='bold', color=arrow_colors[-1])
                ax.text(x, y_display + 0.28, "GOAL", ha='center', va='center',
                       fontsize=8, fontweight='bold', color='#1F5024')
            elif policy_all[x, y] == [-2]:
                ax.text(x, y_display, arrows[-2], ha='center', va='center', 
                       fontsize=20, fontweight='bold', color=arrow_colors[-2])
            else:
                optimal_actions = policy_all[x, y]
                if len(optimal_actions) == 1:
                    action = optimal_actions[0]
                    ax.text(x, y_display, arrows[action], ha='center', va='center', 
                           fontsize=22, fontweight='bold', color=arrow_colors[action])
                else:
                    arrow_text = ' '.join([arrows[action] for action in optimal_actions])
                    ax.text(x, y_display, arrow_text, ha='center', va='center', 
                           fontsize=16, fontweight='bold', color='#2C3E50')
            
            # Display values
            if (x, y) not in env.terminal_states and (x, y) not in env.obstacles:
                max_q = np.max(Q[x, y, :])
                ax.text(x, y_display - 0.25, f"V={V[x,y]:.2f}", ha='center', va='center', 
                       fontsize=8, color='#666666')
                ax.text(x, y_display - 0.35, f"maxQ={max_q:.2f}", ha='center', va='center', 
                       fontsize=7, color='#999999')

    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.5, m - 0.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(n))
    ax.set_yticks(range(m))
    ax.set_title("POLITIQUE OPTIMALE - Q-LEARNING\n(Avec toutes les actions optimales possibles)", 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Legend
    legend_text = "LÉGENDE:\n"
    for action, name in action_names.items():
        legend_text += f"{arrows[action]} = {name}\n"
    legend_text += "→ ↑ = Actions multiples optimales"
    
    ax.text(n + 0.3, m - 0.5, legend_text, fontsize=10, va='top', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save
    os.makedirs("03_Level2_TD/q_learning_results", exist_ok=True)
    policy_filename = "03_Level2_TD/q_learning_results/optimal_policy_q_learning.png"
    plt.savefig(policy_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📸 Politique optimale sauvegardée: {policy_filename}")
    
    plt.show()

def main():
    """
    Main function to run Q-learning on GridWorld
    """
    # Create the 8x4 environment with obstacles
    env = GW(
        grid_width=8,
        grid_height=4,
        terminal_states=[(0, 0), (7, 3)],  # Coins opposés
        obstacle_positions=[(2, 1), (4, 0), (5, 2), (2, 2), (6, 1), (0, 3)],  # Obstacles
        step_penalty=-0.1,
        goal_reward=10,
        obstacle_penalty=-2,  # Pénalité pour heurter un obstacle
        max_steps=100,
        show_agent=True,
        show_goals=True
    )
    
    # Learning parameters
    num_episodes = 1000
    save_gif_episodes = [0, num_episodes//2, num_episodes-1]
    
    # Run Q-learning
    Q, episode_rewards, episode_lengths = q_learning(
        env=env,
        num_episodes=num_episodes,
        gamma=0.9,
        alpha=0.1,
        epsilon=0.4,
        epsilon_decay=0.998,
        min_epsilon=0.02,
        save_gif_episodes=save_gif_episodes
    )
    
    # Extract the optimal policy
    policy, policy_all = extract_optimal_policy(env, Q)
    V = compute_value_function(env, Q)
    
    # Visualizations
    plot_learning_curve(episode_rewards, episode_lengths)
    plot_optimal_policy(env, policy_all, V, Q)
    
    # Display final statistics
    print("\n" + "=" * 50)
    print("STATISTIQUES FINALES")
    print("=" * 50)
    print(f"Récompense moyenne (100 derniers épisodes): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Longueur moyenne (100 derniers épisodes): {np.mean(episode_lengths[-100:]):.2f}")
    print(f"Meilleure récompense: {np.max(episode_rewards)}")
    print(f"Récompense finale: {episode_rewards[-1]}")
    
    # Cleanup
    env.close()

if __name__ == "__main__":
    main()