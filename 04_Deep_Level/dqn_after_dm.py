import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from copy import deepcopy

# =====================================================================
# SOLUTION: DQN AVEC TARGET NETWORK
# Le réseau vise une cible FIXE qui ne change que périodiquement!
# =====================================================================

class DQNetwork(nn.Module):
    """Réseau Q simple (MLP)"""
    def __init__(self, state_dim=2, action_dim=4, hidden_size=64):
        super(DQNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim)
        )
    
    def forward(self, x):
        return self.net(x)


Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class DQNAgentWithTarget:
    """
    ✅ SOLUTION: AVEC TARGET NETWORK
    
    Maintenant nous avons DEUX réseaux:
    1. Q_network (θ): Le réseau principal qu'on entraîne
    2. Target_network (θ⁻): Une copie FIGÉE du réseau principal
    
    Processus:
    1. On calcule Q(s,a) avec Q_network (θ)
    2. On calcule la cible: y = r + γ * max Q_target(s',a') avec Target_network (θ⁻)
    3. On fait gradient descent pour rapprocher Q(s,a) de y
    4. La cible reste FIXE car Target_network ne change pas!
    5. Tous les N updates, on met à jour: θ⁻ ← θ
    
    C'est comme avoir un point de référence stable vers lequel avancer!
    """
    def __init__(self, 
                 state_dim=2,
                 action_dim=4,
                 hidden_size=64,
                 lr=1e-3,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.01,
                 epsilon_decay=0.995,
                 buffer_capacity=10000,
                 batch_size=64,
                 target_update_freq=10):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # ✅ DEUX RÉSEAUX - C'EST LA SOLUTION!
        self.q_network = DQNetwork(state_dim, action_dim, hidden_size).to(self.device)
        self.target_network = DQNetwork(state_dim, action_dim, hidden_size).to(self.device)
        
        # Initialiser le target network avec les mêmes poids
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()  # Pas d'entraînement pour le target!
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.update_count = 0
        self.losses = []
        self.target_updates = []  # Pour tracer les mises à jour du target
        
    def select_action(self, state, eval_mode=False):
        if eval_mode or random.random() > self.epsilon:
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_t)
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_dim)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        transitions = self.replay_buffer.sample(self.batch_size)
        batch = Transition(*zip(*transitions))
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)
        
        # Q(s, a) actuelles avec le réseau principal
        current_q_values = self.q_network(state_batch).gather(1, action_batch).squeeze()
        
        # ✅ SOLUTION ICI: On utilise le TARGET NETWORK pour les cibles!
        with torch.no_grad():
            # Les cibles sont calculées avec le target network qui reste FIXE
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Calcul de la perte et mise à jour DU RÉSEAU PRINCIPAL SEULEMENT
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.losses.append(loss.item())
        self.update_count += 1
        
        # ✅ Mise à jour périodique du target network
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_updates.append(self.update_count)
            print(f"  🎯 Target network mis à jour! (update #{self.update_count})")
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_t).cpu().numpy()[0]


# Environnement GridWorld simplifié
class SimpleGridWorld:
    def __init__(self, size=10, goal_pos=(9, 9), max_steps=100):
        self.size = size
        self.goal_pos = goal_pos
        self.max_steps = max_steps
        self.reset()
    
    def reset(self):
        while True:
            self.agent_pos = [np.random.randint(0, self.size), 
                             np.random.randint(0, self.size)]
            if tuple(self.agent_pos) != self.goal_pos:
                break
        self.step_count = 0
        return np.array(self.agent_pos, dtype=np.float32)
    
    def step(self, action):
        self.step_count += 1
        x, y = self.agent_pos
        
        if action == 0:  # up
            y = max(0, y - 1)
        elif action == 1:  # down
            y = min(self.size - 1, y + 1)
        elif action == 2:  # right
            x = min(self.size - 1, x + 1)
        elif action == 3:  # left
            x = max(0, x - 1)
        
        self.agent_pos = [x, y]
        
        if tuple(self.agent_pos) == self.goal_pos:
            reward = 1.0
            done = True
        else:
            reward = -0.01
            done = False
        
        if self.step_count >= self.max_steps:
            done = True
        
        return np.array(self.agent_pos, dtype=np.float32), reward, done


def train_and_visualize():
    print("="*70)
    print("SOLUTION AU MOVING TARGET PROBLEM")
    print("Version AVEC Target Network")
    print("="*70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    env = SimpleGridWorld(size=10, goal_pos=(9, 9), max_steps=100)
    agent = DQNAgentWithTarget(
        state_dim=2,
        action_dim=4,
        hidden_size=64,
        lr=1e-3,
        gamma=0.99,
        batch_size=64,
        target_update_freq=10  # Mise à jour tous les 10 updates
    )
    
    # États à monitorer
    monitor_state = np.array([5, 5], dtype=np.float32)
    q_values_history = []
    episode_rewards = []
    
    num_episodes = 300
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            episode_reward += reward
            state = next_state
        
        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        q_values_history.append(agent.get_q_values(monitor_state).max())
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Épisode {episode+1}/{num_episodes} | "
                  f"Récompense moyenne: {avg_reward:.2f} | "
                  f"Epsilon: {agent.epsilon:.3f}")
    
    # Visualisation
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Récompenses
    ax = axes[0, 0]
    window = 20
    smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
    ax.plot(smoothed, color='green', linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Récompense moyenne')
    ax.set_title('Convergence STABLE - Avec Target Network')
    ax.grid(True, alpha=0.3)
    
    # 2. Q-values (stable)
    ax = axes[0, 1]
    ax.plot(q_values_history, color='green', alpha=0.7, linewidth=1.5)
    # Marquer les mises à jour du target network
    for update_step in agent.target_updates:
        episode_approx = update_step // 10  # Approximation
        if episode_approx < len(q_values_history):
            ax.axvline(x=episode_approx, color='red', alpha=0.3, linestyle='--', linewidth=1)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Max Q-value')
    ax.set_title('Q-values STABLES (Cible fixe!)')
    ax.text(0.98, 0.02, 'Lignes rouges = mises à jour target', 
            transform=ax.transAxes, ha='right', va='bottom', 
            fontsize=8, style='italic', alpha=0.7)
    ax.grid(True, alpha=0.3)
    
    # 3. Loss
    ax = axes[1, 0]
    if agent.losses:
        window = 50
        smoothed_loss = np.convolve(agent.losses, np.ones(window)/window, mode='valid')
        ax.plot(smoothed_loss, color='green', alpha=0.7)
        ax.set_xlabel('Mise à jour')
        ax.set_ylabel('Loss')
        ax.set_title('Loss d\'entraînement (converge)')
        ax.grid(True, alpha=0.3)
    
    # 4. Schéma explicatif
    ax = axes[1, 1]
    ax.axis('off')
    explanation = """
    ✅ SOLUTION: TARGET NETWORK
    
    Architecture:
    • Q_network (θ): Réseau principal (entraîné)
    • Target_network (θ⁻): Copie figée
    
    Calcul de la cible:
    y = r + γ × max Q_target(s', a')
              ↑
              Réseau FIGÉ pendant N updates!
    
    Avantages:
    ✓ Cible stable → convergence régulière
    ✓ Q-values ne divergent pas
    ✓ Apprentissage plus rapide
    ✓ Meilleure performance finale
    
    Mise à jour:
    Tous les 10 updates: θ⁻ ← θ
    """
    ax.text(0.1, 0.5, explanation, fontsize=11, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig('dqn_with_target_solution.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("ANALYSE DE LA SOLUTION")
    print("="*70)
    print("\n✅ AVEC TARGET NETWORK:")
    print("  1. Q_network (θ) calcule Q(s,a)")
    print("  2. Target_network (θ⁻) calcule y = r + γ * max Q(s',a')")
    print("  3. θ⁻ reste FIXE pendant plusieurs updates")
    print("  4. La cible y ne bouge pas!")
    print("  5. Après N updates: θ⁻ ← θ (synchronisation)")
    print("\n📊 OBSERVATIONS:")
    print("  - Q-values convergent régulièrement (graphique haut-droite)")
    print("  - Lignes rouges = moments de mise à jour du target")
    print("  - Apprentissage stable et efficace")
    print("  - Pas d'oscillations chaotiques")
    print("\n💡 ANALOGIE:")
    print("  Sans target: Essayer d'attraper votre ombre qui bouge")
    print("  Avec target: Marcher vers un point de repère fixe")
    print("="*70)


if __name__ == "__main__":
    train_and_visualize()