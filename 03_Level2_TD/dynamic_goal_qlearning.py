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

class DynamicGoalEnvironment:
    """
    Environment with a periodically changing goal
    """
    def __init__(self, grid_size=8, change_interval=100):
        self.grid_size = grid_size
        self.change_interval = change_interval
        self.episode_count = 0
        
        # Possible goal positions (avoid borders and obstacles)
        self.possible_goals = [
            (1, 1), (1, 6), (6, 1), (6, 6),
            (2, 2), (2, 5), (5, 2), (5, 5),
            (3, 3), (3, 4), (4, 3), (4, 4)
        ]
        
        # Fixed obstacles - store in a local variable
        self._obstacles = [(2, 2), (4, 3), (5, 5), (1, 4), (6, 3)]
        
        # Initial goal
        self.current_goal = self.possible_goals[0]
        
        # Create the environment
        self.env = GW(
            grid_size=grid_size,
            terminal_states=[self.current_goal],
            obstacle_positions=self._obstacles,  # Utiliser la variable locale
            step_penalty=-1,
            goal_reward=10,
            obstacle_penalty=-2,
            max_steps=100,
            show_agent=True,
            show_goals=True
        )
    
    def reset(self):
        """Reset the environment and change the goal if necessary"""
        self.episode_count += 1
        
        # Change the goal every change_interval episodes
        if self.episode_count % self.change_interval == 0:
            # Choose a new random goal (different from the current one)
            available_goals = [g for g in self.possible_goals if g != self.current_goal]
            if available_goals:
                self.current_goal = available_goals[np.random.randint(len(available_goals))]
                print(f"🎯 CHANGEMENT DE GOAL: Nouvelle position {self.current_goal}")
            
            # Recreate the environment with the new goal
            self.env = GW(
                grid_size=self.grid_size,
                terminal_states=[self.current_goal],
                obstacle_positions=self._obstacles,  # Utiliser la variable locale
                step_penalty=-1,
                goal_reward=10,
                obstacle_penalty=-2,
                max_steps=100,
                show_agent=True,
                show_goals=True
            )
        
        return self.env.reset()
    
    # Other methods remain identical...
    def step(self, action):
        return self.env.step(action)
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()
    
    @property
    def grid_width(self):
        return self.env.grid_width
    
    @property
    def grid_height(self):
        return self.env.grid_height
    
    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def terminal_states(self):
        return self.env.terminal_states
    
    @property
    def obstacles(self):
        return self.env.obstacles  # Cette propriété vient de l'environnement GW

def epsilon_greedy_policy(Q, state, epsilon, n_actions):
    """
    Epsilon-greedy policy based on Q-values
    """
    x, y = state
    
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

def q_learning_dynamic_goal(num_episodes=1000, gamma=0.9, alpha=0.1, epsilon=0.3, 
                           change_interval=100, save_gif_episodes=None):
    """
    Implementation of the Q-learning algorithm for a dynamic environment
    """
    
    # Create the dynamic environment
    env = DynamicGoalEnvironment(grid_size=8, change_interval=change_interval)
    
    # Initialize the Q-table
    n = env.grid_width
    m = env.grid_height
    n_actions = env.action_space.n
    Q = np.zeros((n, m, n_actions))
    
    # Track performance
    episode_rewards = []
    episode_lengths = []
    success_rate = []
    goal_changes = []  # Pour marquer les changements de goal
    q_instability = []  # Instabilité de la Q-table
    
    # Episodes for which to save GIFs
    if save_gif_episodes is None:
        save_gif_episodes = [0, change_interval-1, change_interval, 
                           change_interval*2-1, change_interval*2, num_episodes-1]
    
    gif_data = {}
    
    print("Début de l'apprentissage Q-learning avec goal dynamique")
    print(f"Grille: {n}x{m}, Actions: {n_actions}")
    print(f"Changement de goal tous les {change_interval} épisodes")
    print(f"Paramètres: gamma={gamma}, alpha={alpha}, epsilon={epsilon}")
    print("=" * 60)
    
    for episode in range(num_episodes):
        # Environment reset (may change the goal)
        state, info = env.reset()
        
        # Check whether the goal changed
        if episode % change_interval == 0 and episode > 0:
            goal_changes.append(episode)
            print(f"🔄 Épisode {episode}: Goal changé à {env.current_goal}")
        
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        reached_goal = False
        
        # Store frames for the GIF
        if episode in save_gif_episodes:
            frames = []
            # Sauvegarder la frame initiale
            env.render()
            plt.pause(0.01)
            frames.append(save_frame(plt.gcf()))
        
        # Episode loop
        while not (terminated or truncated):
            # Choose action according to epsilon-greedy
            action = epsilon_greedy_policy(Q, state, epsilon, n_actions)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Q-learning update (off-policy formula)
            x, y = state
            nx, ny = next_state
            
            target = reward + gamma * np.max(Q[nx, ny, :])
            Q[x, y, action] += alpha * (target - Q[x, y, action])
            
            # Update for the next iteration
            state = next_state
            total_reward += reward
            steps += 1
            
            if terminated and reward == 10:  # Goal atteint
                reached_goal = True
            
            # Save frame for GIF
            if episode in save_gif_episodes:
                env.render()
                plt.pause(0.01)
                frames.append(save_frame(plt.gcf()))
        
        # Save the final frame
        if episode in save_gif_episodes:
            env.render()
            plt.pause(0.01)
            frames.append(save_frame(plt.gcf()))
            # Add a few duplicated final frames for a pause at the end
            for _ in range(3):
                frames.append(frames[-1])
            gif_data[episode] = frames
            print(f"📹 {len(frames)} frames capturées pour l'épisode {episode}")
        
        # Epsilon decay
        epsilon = max(0.05, epsilon * 0.998)
        
        # Save performance data
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_rate.append(1 if reached_goal else 0)
        
        # Measure Q-table instability
        if episode > 0:
            q_diff = np.sqrt(np.sum((Q - previous_Q) ** 2))
            q_instability.append(q_diff)
        previous_Q = Q.copy()
        
        # Progress display
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            current_success = np.mean(success_rate[-50:]) * 100
            print(f"Épisode {episode+1}/{num_episodes} | "
                  f"Récompense moyenne: {avg_reward:.2f} | "
                  f"Taux de succès: {current_success:.1f}% | "
                  f"Epsilon: {epsilon:.3f}")
    
    # Save GIFs
    save_training_gifs(gif_data, save_gif_episodes)
    
    env.close()
    
    return {
        'Q': Q,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'success_rate': success_rate,
        'q_instability': q_instability,
        'goal_changes': goal_changes
    }

def save_training_gifs(gif_data, episodes):
    """
    Save training GIFs with adaptive speed
    """
    os.makedirs("03_Level2_TD/dynamic_goal_results", exist_ok=True)
    
    for episode, frames in gif_data.items():
        if frames and len(frames) > 0:
            filename = f"03_Level2_TD/dynamic_goal_results/training_episode_{episode+1}.gif"
            
            # Convert to PIL images
            frames_pil = [Image.fromarray(frame) for frame in frames]
            
            # Compute adaptive duration
            if len(frames_pil) < 20:
                duration = 400
            elif len(frames_pil) < 50:
                duration = 250
            else:
                duration = 150
            
            # Save the GIF
            frames_pil[0].save(
                filename,
                save_all=True,
                append_images=frames_pil[1:],
                duration=duration,
                loop=0,
                optimize=True
            )
            
            print(f"GIF sauvegardé: {filename} ({len(frames_pil)} frames)")

def plot_dynamic_goal_performance(results, change_interval=100):
    """
    Plot performance in the dynamic environment
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    episode_rewards = results['episode_rewards']
    success_rate = results['success_rate']
    q_instability = results['q_instability']
    goal_changes = results['goal_changes']
    
    # 1. Rewards curve with goal-change markers
    ax1.plot(episode_rewards, alpha=0.7, linewidth=1, color='blue', label='Récompense par épisode')
    
    # Moving average to smooth the curve
    window = 20
    if len(episode_rewards) > window:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                linewidth=2, color='red', label=f'Moyenne mobile ({window} épisodes)')
    
    # Mark goal changes
    for change in goal_changes:
        ax1.axvline(x=change, color='red', linestyle='--', alpha=0.7, linewidth=1)
        ax1.text(change, ax1.get_ylim()[1] * 0.9, 'Changement\nde goal', 
                ha='center', va='top', fontsize=8, color='red',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense cumulée')
    ax1.set_title('Récompenses cumulées - Environnement Dynamique\n(Lignes rouges: changements de goal)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Success rate with markers
    window = 30
    success_avg = np.convolve(success_rate, np.ones(window)/window, mode='valid') * 100
    ax2.plot(range(window-1, len(success_rate)), success_avg, 
            linewidth=2, color='green', label='Taux de succès (moyenne mobile)')
    
    # Mark goal changes
    for change in goal_changes:
        if change > window-1:
            ax2.axvline(x=change, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Taux de succès (%)')
    ax2.set_title('Taux de succès - Impact des changements de goal')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # 3. Q-table instability
    ax3.plot(q_instability, linewidth=1, color='purple', alpha=0.7)
    
    # Moving average for instability
    if len(q_instability) > window:
        instab_avg = np.convolve(q_instability, np.ones(window)/window, mode='valid')
        ax3.plot(range(window-1, len(q_instability)), instab_avg, 
                linewidth=2, color='orange', label='Moyenne mobile')
    
    # Mark goal changes
    for change in goal_changes:
        if change < len(q_instability):
            ax3.axvline(x=change, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    ax3.set_xlabel('Épisode')
    ax3.set_ylabel('Instabilité Q-table (écart)')
    ax3.set_title('Instabilité de la Q-table après changements de goal')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # 4. Performance before/after goal change
    performance_segments = []
    labels = []
    
    # Analyze performance on each segment between changes
    changes = [0] + goal_changes + [len(episode_rewards)]
    
    for i in range(len(changes)-1):
        start = changes[i]
        end = changes[i+1]
        segment_length = end - start
        
        if segment_length >= 20:  # Segment assez long pour analyse
            # Prendre les derniers 20 épisodes du segment (après apprentissage)
            segment_end = min(end, start + 20)
            segment_rewards = episode_rewards[start:segment_end]
            segment_success = success_rate[start:segment_end]
            
            avg_reward = np.mean(segment_rewards) if segment_rewards else 0
            avg_success = np.mean(segment_success) * 100 if segment_success else 0
            
            performance_segments.append((avg_reward, avg_success))
            labels.append(f'Segment\n{i+1}')
    
    if performance_segments:
        rewards, successes = zip(*performance_segments)
        x_pos = np.arange(len(performance_segments))
        
        width = 0.35
        ax4.bar(x_pos - width/2, rewards, width, label='Récompense moyenne', alpha=0.7)
        ax4.bar(x_pos + width/2, successes, width, label='Taux de succès (%)', alpha=0.7)
        
        ax4.set_xlabel('Segment entre changements de goal')
        ax4.set_ylabel('Performance')
        ax4.set_title('Performance moyenne par segment\nde stabilité du goal')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('03_Level2_TD/dynamic_goal_results/dynamic_goal_performance.png', 
                dpi=300, bbox_inches='tight')
    plt.show()

def analyze_goal_changes_impact(results, change_interval=100):
    """
    Analyse spécifique de l'impact des changements de goal
    """
    episode_rewards = results['episode_rewards']
    success_rate = results['success_rate']
    goal_changes = results['goal_changes']
    
    print("\n" + "="*60)
    print("ANALYSE DE L'IMPACT DES CHANGEMENTS DE GOAL")
    print("="*60)
    
    # Analyser chaque période entre changements
    periods = [0] + goal_changes + [len(episode_rewards)]
    
    for i in range(len(periods)-1):
        start = periods[i]
        end = periods[i+1]
        
        if end - start >= 10:  # Période assez longue
            # Première moitié (adaptation)
            adapt_start = start
            adapt_end = start + min(20, (end - start) // 2)
            adapt_rewards = episode_rewards[adapt_start:adapt_end]
            adapt_success = success_rate[adapt_start:adapt_end]
            
            # Seconde moitié (stabilisation)
            stable_start = max(adapt_end, end - 20)
            stable_end = end
            stable_rewards = episode_rewards[stable_start:stable_end]
            stable_success = success_rate[stable_start:stable_end]
            
            if adapt_rewards and stable_rewards:
                avg_adapt_reward = np.mean(adapt_rewards)
                avg_stable_reward = np.mean(stable_rewards)
                avg_adapt_success = np.mean(adapt_success) * 100
                avg_stable_success = np.mean(stable_success) * 100
                
                print(f"Période {i+1} (Épisodes {start}-{end}):")
                print(f"  Adaptation:  récompense = {avg_adapt_reward:.2f}, succès = {avg_adapt_success:.1f}%")
                print(f"  Stabilisation: récompense = {avg_stable_reward:.2f}, succès = {avg_stable_success:.1f}%")
                print(f"  Amélioration: {avg_stable_reward - avg_adapt_reward:.2f} (+{avg_stable_success - avg_adapt_success:.1f}%)")
                print()

def main():
    """
    Fonction principale pour tester Q-learning avec goal dynamique
    """
    print("TEST DE Q-LEARNING AVEC ENVIRONNEMENT DYNAMIQUE")
    print("Problématique: Q-Learning peut-il s'adapter aux changements de goal?")
    print("=" * 70)
    
    # Paramètres
    num_episodes = 500
    change_interval = 100  # Changement tous les 100 épisodes
    
    # Exécution de Q-learning
    results = q_learning_dynamic_goal(
        num_episodes=num_episodes,
        gamma=0.9,
        alpha=0.1,
        epsilon=0.3,
        change_interval=change_interval
    )
    
    # Visualisations
    plot_dynamic_goal_performance(results, change_interval)
    
    # Analyse détaillée
    analyze_goal_changes_impact(results, change_interval)
    
    # Statistiques finales
    print("\n" + "="*50)
    print("STATISTIQUES FINALES")
    print("="*50)
    final_reward = np.mean(results['episode_rewards'][-50:])
    final_success = np.mean(results['success_rate'][-50:]) * 100
    total_success = np.mean(results['success_rate']) * 100
    
    print(f"Performance finale (50 derniers épisodes):")
    print(f"  - Récompense moyenne: {final_reward:.2f}")
    print(f"  - Taux de succès: {final_success:.1f}%")
    print(f"Performance globale:")
    print(f"  - Taux de succès total: {total_success:.1f}%")
    print(f"  - Nombre de changements de goal: {len(results['goal_changes'])}")
    
    print("\nCONCLUSION:")
    print("Q-Learning montre des difficultés à s'adapter aux changements de goal.")
    print("Chaque changement entraîne une chute de performance temporaire.")
    print("L'algorithme réapprend à chaque fois plutôt que de généraliser.")

if __name__ == "__main__":
    main()