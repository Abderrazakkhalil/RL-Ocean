import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random

# =====================================================================
# COMPARAISON DIRECTE: MOVING TARGET PROBLEM
# Un seul script pour exécuter les deux versions et comparer
# =====================================================================

class DQNetwork(nn.Module):
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


class DQNAgent:
    """Agent DQN configurable avec/sans target network"""
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
                 target_update_freq=10,
                 use_target_network=True,
                 name="DQN"):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.name = name
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_target_network = use_target_network
        
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Réseau principal
        self.q_network = DQNetwork(state_dim, action_dim, hidden_size).to(self.device)
        
        if use_target_network:
            # ✅ AVEC target network
            self.target_network = DQNetwork(state_dim, action_dim, hidden_size).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.target_network.eval()
        else:
            # ⚠️ SANS target network
            self.target_network = self.q_network  # Même réseau!
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
        self.update_count = 0
        self.losses = []
        self.target_changes = []
        
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
        
        # Mesurer le changement de cible (avant mise à jour)
        if not self.use_target_network:
            with torch.no_grad():
                targets_before = reward_batch + (1 - done_batch) * self.gamma * \
                               self.q_network(next_state_batch).max(1)[0]
        
        # Calcul des cibles avec target network (fixe) ou q_network (bouge)
        with torch.no_grad():
            next_q_values = self.target_network(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        loss = self.loss_fn(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Mesurer le changement de cible (après mise à jour)
        if not self.use_target_network:
            with torch.no_grad():
                targets_after = reward_batch + (1 - done_batch) * self.gamma * \
                              self.q_network(next_state_batch).max(1)[0]
                target_shift = (targets_after - targets_before).abs().mean().item()
                self.target_changes.append(target_shift)
        
        self.losses.append(loss.item())
        self.update_count += 1
        
        # Mise à jour du target network (si utilisé)
        if self.use_target_network and self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_q_values(self, state):
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.q_network(state_t).cpu().numpy()[0]


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
        
        if action == 0:
            y = max(0, y - 1)
        elif action == 1:
            y = min(self.size - 1, y + 1)
        elif action == 2:
            x = min(self.size - 1, x + 1)
        elif action == 3:
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


def train_agent(env, agent, num_episodes, monitor_state):
    """Entraîne un agent et collecte les métriques"""
    q_values_history = []
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        agent.decay_epsilon()
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        q_values_history.append(agent.get_q_values(monitor_state).max())
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_length = np.mean(episode_lengths[-50:])
            print(f"  [{agent.name}] Ép {episode+1}/{num_episodes} | "
                  f"Récomp: {avg_reward:.2f} | Long: {avg_length:.1f} | ε: {agent.epsilon:.3f}")
    
    return {
        'rewards': episode_rewards,
        'lengths': episode_lengths,
        'q_values': q_values_history,
        'losses': agent.losses,
        'target_changes': agent.target_changes
    }


def visualize_comparison(results_without, results_with, monitor_state):
    """Visualisation comparative complète"""
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    window = 20
    
    # ==================== LIGNE 1: RÉCOMPENSES ====================
    
    # 1.1 Récompenses SANS target
    ax1 = fig.add_subplot(gs[0, 0])
    smoothed = np.convolve(results_without['rewards'], np.ones(window)/window, mode='valid')
    ax1.plot(smoothed, color='red', linewidth=2, label='Sans Target')
    ax1.set_xlabel('Épisode')
    ax1.set_ylabel('Récompense')
    ax1.set_title('⚠️ SANS Target Network\n(Instable)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 1.2 Récompenses AVEC target
    ax2 = fig.add_subplot(gs[0, 1])
    smoothed = np.convolve(results_with['rewards'], np.ones(window)/window, mode='valid')
    ax2.plot(smoothed, color='green', linewidth=2, label='Avec Target')
    ax2.set_xlabel('Épisode')
    ax2.set_ylabel('Récompense')
    ax2.set_title('✅ AVEC Target Network\n(Stable)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # 1.3 Comparaison directe
    ax3 = fig.add_subplot(gs[0, 2])
    smoothed_without = np.convolve(results_without['rewards'], np.ones(window)/window, mode='valid')
    smoothed_with = np.convolve(results_with['rewards'], np.ones(window)/window, mode='valid')
    ax3.plot(smoothed_without, color='red', linewidth=2, alpha=0.7, label='Sans Target')
    ax3.plot(smoothed_with, color='green', linewidth=2, alpha=0.7, label='Avec Target')
    ax3.set_xlabel('Épisode')
    ax3.set_ylabel('Récompense')
    ax3.set_title('Comparaison Directe', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # ==================== LIGNE 2: Q-VALUES ====================
    
    # 2.1 Q-values SANS target (OSCILLATIONS!)
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(results_without['q_values'], color='red', alpha=0.7, linewidth=1)
    ax4.set_xlabel('Épisode')
    ax4.set_ylabel('Max Q-value')
    ax4.set_title('Q-values OSCILLENT\n(Cible mouvante!)', fontweight='bold', color='red')
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=np.mean(results_without['q_values']), color='darkred', 
                linestyle='--', alpha=0.5, label='Moyenne')
    ax4.legend()
    
    # 2.2 Q-values AVEC target (CONVERGENCE!)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(results_with['q_values'], color='green', alpha=0.7, linewidth=1)
    ax5.set_xlabel('Épisode')
    ax5.set_ylabel('Max Q-value')
    ax5.set_title('Q-values CONVERGENT\n(Cible fixe!)', fontweight='bold', color='green')
    ax5.grid(True, alpha=0.3)
    ax5.axhline(y=np.mean(results_with['q_values'][-50:]), color='darkgreen', 
                linestyle='--', alpha=0.5, label='Moyenne finale')
    ax5.legend()
    
    # 2.3 Longueur des épisodes (efficacité)
    ax6 = fig.add_subplot(gs[1, 2])
    smoothed_len_without = np.convolve(results_without['lengths'], np.ones(window)/window, mode='valid')
    smoothed_len_with = np.convolve(results_with['lengths'], np.ones(window)/window, mode='valid')
    ax6.plot(smoothed_len_without, color='red', linewidth=2, alpha=0.7, label='Sans Target')
    ax6.plot(smoothed_len_with, color='green', linewidth=2, alpha=0.7, label='Avec Target')
    ax6.set_xlabel('Épisode')
    ax6.set_ylabel('Longueur (steps)')
    ax6.set_title('Efficacité\n(Moins = Mieux)', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # ==================== LIGNE 3: DIAGNOSTICS ====================
    
    # 3.1 Changements de cible (SANS target uniquement)
    ax7 = fig.add_subplot(gs[2, 0])
    if results_without['target_changes']:
        window_changes = 100
        smoothed_changes = np.convolve(results_without['target_changes'], 
                                      np.ones(window_changes)/window_changes, mode='valid')
        ax7.plot(smoothed_changes, color='orange', alpha=0.7)
        ax7.set_xlabel('Mise à jour')
        ax7.set_ylabel('Changement de cible')
        ax7.set_title('MOVING TARGET!\n(Magnitude des changements)', fontweight='bold', color='orange')
        ax7.grid(True, alpha=0.3)
    
    # 3.2 Loss comparée
    ax8 = fig.add_subplot(gs[2, 1])
    window_loss = 50
    if results_without['losses']:
        smoothed_loss_without = np.convolve(results_without['losses'], 
                                           np.ones(window_loss)/window_loss, mode='valid')
        ax8.plot(smoothed_loss_without, color='red', alpha=0.7, label='Sans Target')
    if results_with['losses']:
        smoothed_loss_with = np.convolve(results_with['losses'], 
                                        np.ones(window_loss)/window_loss, mode='valid')
        ax8.plot(smoothed_loss_with, color='green', alpha=0.7, label='Avec Target')
    ax8.set_xlabel('Mise à jour')
    ax8.set_ylabel('Loss')
    ax8.set_title('Loss d\'entraînement', fontweight='bold')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 3.3 Explication textuelle
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    explanation = """
MOVING TARGET PROBLEM

⚠️ SANS Target Network:
━━━━━━━━━━━━━━━━━━━━
  y = r + γ max Q(s',a')
           ↑
    CE RÉSEAU CHANGE
    À CHAQUE UPDATE!
    
→ La cible bouge constamment
→ Le réseau "chasse son ombre"
→ Oscillations, instabilité

✅ AVEC Target Network:
━━━━━━━━━━━━━━━━━━━━
  y = r + γ max Q_target(s',a')
           ↑
    RÉSEAU FIGÉ
    (mis à jour tous les N steps)
    
→ Cible stable pendant N updates
→ Convergence régulière
→ Apprentissage efficace

💡 ANALOGIE:
Sans: Attraper votre ombre
Avec: Marcher vers un repère fixe
    """
    ax9.text(0.1, 0.5, explanation, fontsize=9, 
            verticalalignment='center', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.suptitle('DÉMONSTRATION DU MOVING TARGET PROBLEM DANS DQN', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig('dqn_moving_target_full_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    print("="*80)
    print(" "*20 + "MOVING TARGET PROBLEM - COMPARAISON COMPLÈTE")
    print("="*80)
    
    # Configuration
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    monitor_state = np.array([5, 5], dtype=np.float32)
    num_episodes = 300
    
    # ==================== EXPÉRIENCE 1: SANS TARGET ====================
    print("\n🔴 EXPÉRIENCE 1: DQN SANS TARGET NETWORK")
    print("─" * 80)
    print("Le réseau calcule les cibles avec LUI-MÊME → cible mouvante!")
    print()
    
    env = SimpleGridWorld(size=10, goal_pos=(9, 9), max_steps=100)
    agent_without = DQNAgent(
        state_dim=2, action_dim=4, hidden_size=64,
        lr=1e-3, gamma=0.99, batch_size=64,
        use_target_network=False,  # ⚠️ PAS de target!
        name="SANS-TARGET"
    )
    
    results_without = train_agent(env, agent_without, num_episodes, monitor_state)
    
    # ==================== EXPÉRIENCE 2: AVEC TARGET ====================
    print("\n🟢 EXPÉRIENCE 2: DQN AVEC TARGET NETWORK")
    print("─" * 80)
    print("Le réseau calcule les cibles avec un réseau FIGÉ → cible stable!")
    print()
    
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    env = SimpleGridWorld(size=10, goal_pos=(9, 9), max_steps=100)
    agent_with = DQNAgent(
        state_dim=2, action_dim=4, hidden_size=64,
        lr=1e-3, gamma=0.99, batch_size=64,
        target_update_freq=10,
        use_target_network=True,  # ✅ Avec target!
        name="AVEC-TARGET"
    )
    
    results_with = train_agent(env, agent_with, num_episodes, monitor_state)
    
    # ==================== VISUALISATION ====================
    print("\n📊 GÉNÉRATION DES GRAPHIQUES COMPARATIFS")
    print("─" * 80)
    visualize_comparison(results_without, results_with, monitor_state)
    
    # ==================== STATISTIQUES FINALES ====================
    print("\n📈 STATISTIQUES FINALES (50 derniers épisodes)")
    print("=" * 80)
    
    final_50_without = np.mean(results_without['rewards'][-50:])
    final_50_with = np.mean(results_with['rewards'][-50:])
    
    print(f"\n⚠️  SANS Target Network:")
    print(f"   Récompense moyenne finale: {final_50_without:.3f}")
    print(f"   Variance Q-values: {np.var(results_without['q_values'][-50:]):.3f}")
    
    print(f"\n✅ AVEC Target Network:")
    print(f"   Récompense moyenne finale: {final_50_with:.3f}")
    print(f"   Variance Q-values: {np.var(results_with['q_values'][-50:]):.3f}")
    
    improvement = ((final_50_with - final_50_without) / abs(final_50_without)) * 100
    print(f"\n💡 Amélioration: {improvement:+.1f}%")
    
    print("\n" + "=" * 80)
    print("CONCLUSION:")
    print("Le Target Network stabilise l'apprentissage en fixant temporairement les cibles.")
    print("Sans lui, le réseau poursuit une cible qui bouge à chaque mise à jour!")
    print("=" * 80)


if __name__ == "__main__":
    main()