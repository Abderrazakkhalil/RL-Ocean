import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

# =====================================================================
# PROBLÈME: DQN SANS TARGET NETWORK
# Le réseau essaie d'atteindre une cible qui CHANGE à chaque mise à jour!
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


class DQNAgentWithoutTarget:
    """
    ⚠️ PROBLÈME: PAS DE TARGET NETWORK
    
    À chaque mise à jour:
    1. On calcule Q(s,a) avec le réseau actuel
    2. On calcule la cible: y = r + γ * max Q(s',a') avec LE MÊME réseau
    3. On fait un gradient descent pour rapprocher Q(s,a) de y
    4. MAIS maintenant Q(s',a') a changé aussi!
    5. La cible que nous poursuivons bouge constamment!
    
    C'est comme essayer d'attraper votre propre ombre qui bouge quand vous bougez!
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
                 batch_size=64):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        
        # Epsilon-greedy
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # UN SEUL RÉSEAU - C'EST LE PROBLÈME!
        self.q_network = DQNetwork(state_dim, action_dim, hidden_size).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.losses = []
        self.target_changes = []  # Pour tracer les changements de cible
        
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
        
        # Q(s, a) actuelles
        current_q_values = self.q_network(state_batch).gather(1, action_batch).squeeze()
        
        # ⚠️ PROBLÈME ICI: On utilise le MÊME réseau pour les cibles!
        # Avant la mise à jour, calculons les cibles
        with torch.no_grad():
            targets_before = reward_batch + (1 - done_batch) * self.gamma * \
                           self.q_network(next_state_batch).max(1)[0]
        
        # Calcul de la perte et mise à jour
        with torch.no_grad():
            next_q_values = self.q_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # ⚠️ APRÈS la mise à jour, les cibles ont changé!
        with torch.no_grad():
            targets_after = reward_batch + (1 - done_batch) * self.gamma * \
                          self.q_network(next_state_batch).max(1)[0]
            target_shift = (targets_after - targets_before).abs().mean().item()
            self.target_changes.append(target_shift)
        
        self.losses.append(loss.item())
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
    print("DÉMONSTRATION DU MOVING TARGET PROBLEM")
    print("Version SANS Target Network")
    print("="*70)
    
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    env = SimpleGridWorld(size=10, goal_pos=(9, 9), max_steps=100)
    agent = DQNAgentWithoutTarget(
        state_dim=2,
        action_dim=4,
        hidden_size=64,
        lr=1e-3,
        gamma=0.99,
        batch_size=64
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
    ax.plot(smoothed, color='red', linewidth=2)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Récompense moyenne')
    ax.set_title('Convergence INSTABLE - Sans Target Network')
    ax.grid(True, alpha=0.3)
    
    # 2. Q-values (oscillations)
    ax = axes[0, 1]
    ax.plot(q_values_history, color='red', alpha=0.7, linewidth=1.5)
    ax.set_xlabel('Épisode')
    ax.set_ylabel('Max Q-value')
    ax.set_title('Q-values OSCILLENT (Moving Target!)')
    ax.grid(True, alpha=0.3)
    
    # 3. Changements de cible
    ax = axes[1, 0]
    if agent.target_changes:
        ax.plot(agent.target_changes, color='orange', alpha=0.6)
        ax.set_xlabel('Mise à jour')
        ax.set_ylabel('Changement de cible')
        ax.set_title('Magnitude des changements de cible après chaque update')
        ax.grid(True, alpha=0.3)
    
    # 4. Loss
    ax = axes[1, 1]
    if agent.losses:
        window = 50
        smoothed_loss = np.convolve(agent.losses, np.ones(window)/window, mode='valid')
        ax.plot(smoothed_loss, color='red', alpha=0.7)
        ax.set_xlabel('Mise à jour')
        ax.set_ylabel('Loss')
        ax.set_title('Loss d\'entraînement')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dqn_without_target_problem.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print("ANALYSE DU PROBLÈME")
    print("="*70)
    print("\n⚠️  SANS TARGET NETWORK:")
    print("  1. Le réseau Q calcule: y = r + γ * max Q(s',a')")
    print("  2. On fait un gradient descent pour Q(s,a) → y")
    print("  3. MAIS Q change pendant l'entraînement!")
    print("  4. Donc y change aussi → cible mouvante!")
    print("  5. Le réseau poursuit une cible qui bouge constamment")
    print("\n📊 OBSERVATIONS:")
    print("  - Q-values oscillent fortement (graphique haut-droite)")
    print("  - Convergence lente et instable")
    print("  - Les cibles changent après chaque mise à jour")
    print("="*70)


if __name__ == "__main__":
    train_and_visualize()