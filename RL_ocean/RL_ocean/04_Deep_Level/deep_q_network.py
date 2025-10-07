import numpy as np 
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.animation import FuncAnimation, PillowWriter
import io
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Add parent directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the environment from the renamed package
from GridWorld.grid_env import GW

class DeepQNetwork(nn.Module):
    """
    Neural network for deep Q-learning
    Architecture: States -> Hidden layer -> Action Q-values
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        print("Architecture du reseau de neurones:")
        print(f"   Input: {input_size} neurones (etat)")
        print(f"   Hidden: {hidden_size} neurones")
        print(f"   Output: {output_size} neurones (actions Q-values)")
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DeepQLearningAgent:
    """
    Agent using a neural network for deep Q-learning
    """
    def __init__(self, state_size, action_size, hidden_size=128, lr=0.001, gamma=0.99, epsilon=1.0):
        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.memory = deque(maxlen=2000)
        self.batch_size = 32
        
        # Device (CPU or GPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device utilise: {self.device}")
        
        # Neural networks
        self.q_network = DeepQNetwork(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DeepQNetwork(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Copier les poids initiaux
        self.update_target_network()
        
        print("Agent Deep Q-Learning initialise:")
        print(f"   State size: {state_size}")
        print(f"   Action size: {action_size}")
        print(f"   Learning rate: {lr}")
        print(f"   Gamma: {gamma}")
        print(f"   Epsilon initial: {epsilon}")
        
    def update_target_network(self):
        """Update the target network with the weights of the main network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in the replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Select an action using the epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            # Exploration: random action
            return random.randrange(self.action_size)
        else:
            # Exploitation: best action according to the network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return np.argmax(q_values.cpu().data.numpy())
            
    def replay(self):
        """Train the network on a batch of experiences"""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample a random batch
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Target Q-values
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute the loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()

def state_to_features(state, grid_size=6):
    """
    Convert the state (x, y) into features for the neural network
    One-hot encoding of the position
    """
    x, y = state
    features = np.zeros(grid_size * grid_size)
    position_index = x * grid_size + y
    features[position_index] = 1.0
    return features

def save_frame(fig):
    """
    Save a frame from a matplotlib figure with better quality
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    img = Image.open(buf)
    return np.array(img)

def deep_q_learning(env, num_episodes=1000, gamma=0.9, alpha=0.1, epsilon=0.1, 
                   epsilon_decay=0.995, min_epsilon=0.01, save_gif_episodes=None):
    """
    Implementation of the Deep Q-learning algorithm for GridWorld
    """
    
    # Agent configuration
    grid_size = env.grid_width
    state_size = grid_size * grid_size  # Encodage one-hot
    action_size = env.action_space.n
    hidden_size = 64
    
    agent = DeepQLearningAgent(
        state_size=state_size,
        action_size=action_size,
        hidden_size=hidden_size,
        lr=alpha,
        gamma=gamma,
        epsilon=epsilon
    )
    
    # Track performance
    episode_rewards = []
    episode_lengths = []
    episode_losses = []
    success_rate = []
    
    # Episodes for which to save GIFs
    if save_gif_episodes is None:
        save_gif_episodes = [0, num_episodes//2, num_episodes-1]
    
    gif_data = {}
    
    print("Debut de l'apprentissage Deep Q-learning...")
    print(f"Grille: {env.grid_width}x{env.grid_height}, Actions: {action_size}")
    print(f"Obstacles: {env.obstacles}")
    print(f"Terminal states: {env.terminal_states}")
    print(f"Parametres: gamma={gamma}, alpha={alpha}, epsilon={epsilon}")
    print("=" * 50)
    
    for episode in range(num_episodes):
        # Environment reset
        state, info = env.reset()
        state_features = state_to_features(state, grid_size)
        
        # PROTECTION: Ensure the initial state is NOT an obstacle
        max_retries = 10
        retry_count = 0
        while tuple(state) in env.obstacles and retry_count < max_retries:
            print(f"Agent initialise sur obstacle {tuple(state)}, reessai {retry_count+1}/{max_retries}")
            state, info = env.reset()
            state_features = state_to_features(state, grid_size)
            retry_count += 1
        
        if retry_count >= max_retries:
            print("ERREUR: Impossible d'initialiser l'agent hors des obstacles")
            safe_positions = [(x, y) for x in range(grid_size) for y in range(grid_size) 
                            if (x, y) not in env.obstacles and (x, y) not in env.terminal_states]
            if safe_positions:
                state = np.array(safe_positions[0])
                state_features = state_to_features(state, grid_size)
                print(f"Agent force a la position safe: {state}")
            else:
                print("Aucune position safe disponible!")
                continue
        
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        reached_goal = False
        
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
            action = agent.act(state_features)
            
            # Execute action
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state_features = state_to_features(next_state, grid_size)
            
            # Store experience
            agent.remember(state_features, action, reward, next_state_features, terminated)
            
            # Training
            loss = agent.replay()
            if loss is not None:
                episode_losses.append(loss)
            
            # Update for the next iteration
            state_features = next_state_features
            total_reward += reward
            steps += 1
            
            if terminated and reward == 10:
                reached_goal = True
            
            # Save frame for GIF
            if episode in save_gif_episodes:
                env.render()
                plt.pause(0.01)
                frames.append(save_frame(plt.gcf()))
        
        # Periodically update the target network
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Save the final frame
        if episode in save_gif_episodes:
            env.render()
            plt.pause(0.01)
            frames.append(save_frame(plt.gcf()))
            # Add a few duplicated final frames for a pause at the end
            for _ in range(3):
                frames.append(frames[-1])
            gif_data[episode] = frames
            print(f"{len(frames)} frames capturees pour l'episode {episode+1}")
        
        # Save performance data
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        success_rate.append(1 if reached_goal else 0)
        
        # Progress display
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            current_success = np.mean(success_rate[-100:]) * 100
            memory_usage = len(agent.memory)
            
            print(f"Episode {episode+1}/{num_episodes} | "
                  f"Recompense moyenne: {avg_reward:.2f} | "
                  f"Longueur moyenne: {avg_length:.1f} | "
                  f"Taux de succes: {current_success:.1f}% | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Memory: {memory_usage}")
    
    # Save GIFs
    save_training_gifs(gif_data, save_gif_episodes)
    
    env.close()
    
    return {
        'agent': agent,
        'episode_rewards': episode_rewards,
        'episode_lengths': episode_lengths,
        'episode_losses': episode_losses,
        'success_rate': success_rate
    }

def save_training_gifs(gif_data, episodes):
    """
    Save training GIFs with adaptive speed
    """
    os.makedirs("04_Deep_Level/deep_q_learning_results", exist_ok=True)
    
    for episode, frames in gif_data.items():
        if frames and len(frames) > 0:
            filename = f"04_Deep_Level/deep_q_learning_results/training_episode_{episode+1}.gif"
            
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
            
            print(f"GIF sauvegarde: {filename} ({len(frames_pil)} frames, {duration}ms/frame)")

def plot_learning_curve(episode_rewards, episode_lengths, episode_losses, success_rate):
    """
    Plot learning curves for Deep Q-Learning
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Rewards curve
    ax1.plot(episode_rewards, alpha=0.7, linewidth=0.8, color='blue')
    
    # Moving average to smooth the curve
    window = min(100, len(episode_rewards) // 10)
    if window > 0:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 
                linewidth=2, color='red', label=f'Moyenne mobile ({window} episodes)')
    
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Recompense totale')
    ax1.set_title('Courbe d\'apprentissage - Recompenses par episode')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 2. Episode lengths curve
    ax2.plot(episode_lengths, alpha=0.7, linewidth=0.8, color='green')
    
    if window > 0:
        moving_avg_lengths = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), moving_avg_lengths, 
                linewidth=2, color='orange', label=f'Moyenne mobile ({window} episodes)')
    
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Longueur de l\'episode')
    ax2.set_title('Longueur des episodes')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 3. Loss curve
    if episode_losses:
        # Take a moving average for the loss
        loss_window = min(100, len(episode_losses) // 10)
        if loss_window > 0:
            loss_moving_avg = np.convolve(episode_losses, np.ones(loss_window)/loss_window, mode='valid')
            ax3.plot(range(loss_window-1, len(episode_losses)), loss_moving_avg, 
                    linewidth=2, color='purple')
        
        ax3.set_xlabel('Step d\'entrainement')
        ax3.set_ylabel('Loss')
        ax3.set_title('Loss du reseau de neurones')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
    
    # 4. Success rate
    if len(success_rate) > window:
        success_avg = np.convolve(success_rate, np.ones(window)/window, mode='valid') * 100
        ax4.plot(range(window-1, len(success_rate)), success_avg, 
                linewidth=2, color='green', label='Taux de succes')
    
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Taux de succes (%)')
    ax4.set_title('Taux de succes')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 100)
    ax4.legend()
    
    plt.tight_layout()
    
    # Save
    os.makedirs("04_Deep_Level/deep_q_learning_results", exist_ok=True)
    plt.savefig("04_Deep_Level/deep_q_learning_results/deep_learning_curves.png", dpi=300, bbox_inches='tight')
    plt.show()

def analyze_neural_predictions(agent, env):
    """
    Analyze neural network predictions for a few states
    """
    grid_size = env.grid_width
    
    print("ANALYSE DES PREDICTIONS DU RESEAU DE NEURONES")
    print("=" * 50)
    
    # Test on a few representative states
    test_states = [(0, 0), (2, 2), (grid_size-1, grid_size-1), (1, 1)]
    
    for state in test_states:
        features = state_to_features(state, grid_size)
        state_tensor = torch.FloatTensor(features).unsqueeze(0).to(agent.device)
        
        with torch.no_grad():
            q_values = agent.q_network(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]
            
        print(f"Etat {state}:")
        print(f"  Q-values: {[f'{q:.3f}' for q in q_values_np]}")
        best_action = np.argmax(q_values_np)
        action_names = ["HAUT", "BAS", "DROITE", "GAUCHE"]
        print(f"  Meilleure action: {action_names[best_action]} (valeur: {q_values_np[best_action]:.3f})")
        print()

def plot_neural_analysis(agent, env):
    """
    Create a visualization of neural network predictions
    """
    grid_size = env.grid_width
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    action_names = ["HAUT", "BAS", "DROITE", "GAUCHE"]
    colors = ['red', 'blue', 'green', 'orange']
    
    # Analyze Q-values for each state
    all_q_values = []
    
    for x in range(grid_size):
        for y in range(grid_size):
            if (x, y) not in env.obstacles and (x, y) not in env.terminal_states:
                features = state_to_features((x, y), grid_size)
                state_tensor = torch.FloatTensor(features).unsqueeze(0).to(agent.device)
                
                with torch.no_grad():
                    q_values = agent.q_network(state_tensor)
                    q_values_np = q_values.cpu().numpy()[0]
                    all_q_values.append(q_values_np)
    
    if all_q_values:
        all_q_values = np.array(all_q_values)
        
        # 1. Distribution of Q-values
        for action in range(4):
            axes[0].hist(all_q_values[:, action], bins=20, alpha=0.7, 
                        label=action_names[action], color=colors[action])
        axes[0].set_xlabel('Q-value')
        axes[0].set_ylabel('Frequence')
        axes[0].set_title('Distribution des Q-values par action')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Mean Q-values per action
        mean_q_values = np.mean(all_q_values, axis=0)
        bars = axes[1].bar(action_names, mean_q_values, color=colors, alpha=0.7)
        axes[1].set_ylabel('Q-value moyenne')
        axes[1].set_title('Q-values moyennes par action')
        for bar, value in zip(bars, mean_q_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 3. Variance of Q-values
        std_q_values = np.std(all_q_values, axis=0)
        bars = axes[2].bar(action_names, std_q_values, color=colors, alpha=0.7)
        axes[2].set_ylabel('Ecart-type des Q-values')
        axes[2].set_title('Variabilite des Q-values par action')
        for bar, value in zip(bars, std_q_values):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Correlation between Q-values of actions
        correlation_matrix = np.corrcoef(all_q_values.T)
        im = axes[3].imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        axes[3].set_xticks(range(4))
        axes[3].set_yticks(range(4))
        axes[3].set_xticklabels(action_names)
        axes[3].set_yticklabels(action_names)
        axes[3].set_title('Correlation entre les Q-values des actions')
        
        # Ajouter les valeurs dans la matrice
        for i in range(4):
            for j in range(4):
                text = axes[3].text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                  ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig("04_Deep_Level/deep_q_learning_results/neural_network_analysis.png", 
                dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """
    Fonction principale pour executer Deep Q-learning sur GridWorld
    """
    # Creation de l'environnement 6x6 avec obstacles
    env = GW(
        grid_size=6,
        terminal_states=[(0, 0), (5, 5)],
        obstacle_positions=[(1, 1), (2, 2), (4, 3), (3, 4)],
        step_penalty=-1,
        goal_reward=10,
        obstacle_penalty=-2,
        max_steps=100,
        show_agent=True,
        show_goals=True
    )
    
    # Parametres d'apprentissage
    num_episodes = 800
    save_gif_episodes = [0, num_episodes//2, num_episodes-1]
    
    # Execution de Deep Q-learning
    results = deep_q_learning(
        env=env,
        num_episodes=num_episodes,
        gamma=0.99,
        alpha=0.001,  # Learning rate plus petit pour Deep RL
        epsilon=0.3,
        epsilon_decay=0.995,
        min_epsilon=0.01,
        save_gif_episodes=save_gif_episodes
    )
    
    # Visualisations
    plot_learning_curve(
        results['episode_rewards'],
        results['episode_lengths'],
        results['episode_losses'],
        results['success_rate']
    )
    
    # Analyse du reseau de neurones
    analyze_neural_predictions(results['agent'], env)
    plot_neural_analysis(results['agent'], env)
    
    # Affichage des statistiques finales
    print("\n" + "=" * 50)
    print("STATISTIQUES FINALES - DEEP Q-LEARNING")
    print("=" * 50)
    print(f"Recompense moyenne (100 derniers episodes): {np.mean(results['episode_rewards'][-100:]):.2f}")
    print(f"Longueur moyenne (100 derniers episodes): {np.mean(results['episode_lengths'][-100:]):.2f}")
    print(f"Taux de succes final: {np.mean(results['success_rate'][-100:]) * 100:.1f}%")
    print(f"Meilleure recompense: {np.max(results['episode_rewards'])}")
    print(f"Recompense finale: {results['episode_rewards'][-1]}")
    print(f"Taille de la memoire de replay: {len(results['agent'].memory)}")
    print(f"Epsilon final: {results['agent'].epsilon:.3f}")
    
    print("\nCOMPARAISON AVEC Q-LEARNING CLASSIQUE:")
    print("Avantage: Pas de table Q explicite, economie memoire")
    print("Avantage: Generalisation possible a des etats non vus")
    print("Defi: Plus complexe a entrainer, hyperparametres sensibles")
    print("Defi: Necessite plus de donnees et de calcul")
    
    # Nettoyage
    env.close()

if __name__ == "__main__":
    # Verification de PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    main()