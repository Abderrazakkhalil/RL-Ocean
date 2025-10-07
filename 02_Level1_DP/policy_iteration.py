import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from PI_env import PI
import os

def policy_evaluation(env, policy, V, gamma=0.9, theta=1e-3, max_iterations=100):
    """
    Evaluate a policy until convergence (iterative version)
    """
    print("  Évaluation de la politique en cours...")
    
    for i in range(max_iterations):
        delta = 0
        V_prev = V.copy()
        
        for x in range(4):
            for y in range(4):
                state = np.array([x, y])
                
                # Terminal states - no update
                if np.array_equal(state, [0, 0]) or np.array_equal(state, [3, 3]):
                    continue
                
                # Get the action according to the current policy
                action = policy[x, y]
                next_state, reward, terminated, _, _ = env.step_state(state, action)
                nx, ny = next_state
                
                # Value update (Bellman equation for fixed policy)
                new_value = reward + gamma * V_prev[nx, ny]
                delta = max(delta, abs(new_value - V[x, y]))
                V[x, y] = new_value
        
        if delta < theta:
            print(f"  Évaluation convergée après {i+1} itérations (delta={delta:.6f})")
            break
    
    return V


def policy_iteration(env, V=None, policy=None, gamma=0.9, theta=1e-3, max_depth=20, current_iter=0):
    """
    Implement Policy Iteration with recursion and using extract_policy
    """
    if current_iter == 0:
        print("DÉBUT DE POLICY ITERATION (Version Récursive)")
        print("=" * 50)
        print(f"Paramètres: gamma={gamma}, theta={theta}")
        print(f"États terminaux: (0,0) et (3,3)")
        print("=" * 50)
    
    # Initialization on the first call
    if V is None:
        V = np.zeros((4, 4))
    if policy is None:
        # Random initial policy
        policy = np.random.randint(0, 4, (4, 4))
        policy[0, 0] = -1  # Terminal states
        policy[3, 3] = -1

    print(f"\n*** ITÉRATION {current_iter + 1} ***")
    print(f"Politique actuelle:")
    print(policy)
    
    # === PHASE 1: ÉVALUATION ===
    print("\n[PHASE 1 - ÉVALUATION]")
    old_V = V.copy()
    V = policy_evaluation(env, policy, V, gamma, theta)
    
    print(f"Valeurs après évaluation:")
    print(V.round(3))
    
    # Display
    env.render(V=V, iteration=current_iter + 1)
    
    # === PHASE 2: AMÉLIORATION ===
    print("\n[PHASE 2 - AMÉLIORATION]")
    # Use extract_policy for improvement
    new_policy = extract_policy(env, V, gamma, verbose=False)
    
    print(f"Nouvelle politique proposée:")
    print(new_policy)
    
    # === STOP CONDITION ===
    policy_stable = np.array_equal(policy, new_policy)
    
    if policy_stable:
        print(f"\n🎉 CONVERGENCE ATTEINTE à l'itération {current_iter + 1}!")
        print("La politique est stable - pas de changement depuis l'itération précédente")
        print(f"Valeurs finales optimales:")
        print(V.round(3))
        print(f"Politique optimale:")
        print(new_policy)
        
        # Save the last iteration image
        save_iteration_plot(env, V, current_iter + 1, "final_iteration")
        
        return V, new_policy
    elif current_iter >= max_depth - 1:
        print(f"\n⚠️  Maximum d'itérations ({max_depth}) atteint")
        print("Retour des dernières valeurs et politique")
        
        # Save the last iteration image
        save_iteration_plot(env, V, current_iter + 1, "final_iteration_max_depth")
        
        return V, new_policy
    else:
        print(f"➡️  Politique a changé, poursuite de l'algorithme...")
        print("=" * 40)
        
        # RECURSIVE CALL with the new policy
        return policy_iteration(env, V, new_policy, gamma, theta, max_depth, current_iter + 1)


def extract_policy(env, V, gamma=0.9, verbose=True):
    """
    Extract the optimal policy from the V table
    (Used for policy improvement)
    KEEP argmax for the Policy Iteration algorithm
    """
    if verbose:
        print("EXTRACTION DE LA POLITIQUE OPTIMALE")
        print("=" * 40)
    
    policy = np.zeros((4, 4), dtype=int)
    action_names = ["^", "v", ">", "<"]
    
    for x in range(4):
        for y in range(4):
            state = np.array([x, y])
            if np.array_equal(state, [0, 0]) or np.array_equal(state, [3, 3]):
                policy[x, y] = -1
                if verbose:
                    print(f"  État terminal ({x},{y}): aucune action")
                continue
                
            values = []
            action_details = []
            
            for action in range(4):
                next_state, reward, _, _, _ = env.step_state(state, action)
                nx, ny = next_state
                action_value = reward + gamma * V[nx, ny]
                values.append(action_value)
                
                if verbose:
                    action_details.append(f"{action_names[action]}: {reward:.1f} + {gamma:.1f}×{V[nx, ny]:.3f} = {action_value:.3f}")
            
            # KEEP argmax for the algorithm
            best_action = np.argmax(values)
            policy[x, y] = best_action
            
            if verbose:
                print(f"  État ({x},{y}):")
                for detail in action_details:
                    print(f"    {detail}")
                print(f"    ➤ Action optimale: {action_names[best_action]}")
    
    if verbose:
        print(f"Politique extraite:")
        print(policy)
        print("=" * 40)
    
    return policy


def extract_all_optimal_actions(env, V, gamma=0.9):
    """
    Extract ALL optimal actions possible for the final display
    (Only for the final visualization)
    """
    print("EXTRACTION DE TOUTES LES ACTIONS OPTIMALES POSSIBLES")
    print("=" * 50)
    
    policy_all = np.empty((4, 4), dtype=object)
    action_names = ["^", "v", ">", "<"]
    
    for x in range(4):
        for y in range(4):
            state = np.array([x, y])
            if np.array_equal(state, [0, 0]) or np.array_equal(state, [3, 3]):
                policy_all[x, y] = [-1]  # List with -1 for terminal state
                continue
                
            values = []
            
            for action in range(4):
                next_state, reward, _, _, _ = env.step_state(state, action)
                nx, ny = next_state
                action_value = reward + gamma * V[nx, ny]
                values.append(action_value)
            
            # Find ALL optimal actions
            max_value = max(values)
            best_actions = [action for action, value in enumerate(values) if abs(value - max_value) < 1e-6]
            
            policy_all[x, y] = best_actions
            
            print(f"  État ({x},{y}): {[action_names[a] for a in best_actions]}")
    
    return policy_all


def save_iteration_plot(env, V, iteration, filename_prefix):
    """
    Save the image of the iteration
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.set_xlim(-0.5, env.grid_width - 0.5)
    ax.set_ylim(-0.5, env.grid_height - 0.5)
    ax.set_aspect('equal')
    
    ax.set_xticks(range(env.grid_width))
    ax.set_yticks(range(env.grid_height))
    ax.set_xticklabels([str(i) for i in range(env.grid_width)], fontsize=10)
    ax.set_yticklabels([str(env.grid_height-1-i) for i in range(env.grid_height)], fontsize=10)
    ax.grid(True, alpha=0.3, color='#CCCCCC')
    ax.set_facecolor('#FAFAFA')

    for x in range(env.grid_width):
        for y in range(env.grid_height):
            y_display = env.grid_height - 1 - y
            
            if (x, y) in env.terminal_states:
                color = "#92BD98"
                border_color = "#1F5024"
                border_width = 3
            else:
                color = "#E3F2FD"
                border_color = "#1C4B6F"
                border_width = 2

            rect = plt.Rectangle((x - 0.4, y_display - 0.4), 0.8, 0.8,
                               facecolor=color, edgecolor=border_color, 
                               linewidth=border_width, alpha=0.9)
            ax.add_patch(rect)
            # Mark goals
            if (x, y) in env.terminal_states:
                ax.text(x, y_display, "GOAL", ha='center', va='center', 
                            fontsize=9, fontweight='bold', color= "#1F5024")
       
            if V is not None:
                value = V[x, y]
                if (x, y) in env.terminal_states:
                    text_color = '#E17055'
                    font_weight = 'bold'
                else:
                    text_color = '#2D3436'
                    font_weight = 'normal'
                if (x, y) in env.terminal_states :
                        continue
                else :
                        ax.text(x, y_display, f"{value:.2f}", ha='center', va='center', 
                               color=text_color, fontsize=12, fontweight=font_weight,
                               bbox=dict(boxstyle="round,pad=0.1", facecolor='white', 
                                       edgecolor=border_color, linewidth=1, alpha=0.8))
                
    title = f"Policy Iteration - Itération {iteration}"
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15, color='#2C3E50')
    
    # Create the folder if it does not exist
    os.makedirs("02_Level1_DP/results", exist_ok=True)
    
    # Save the image
    filename = f"02_Level1_DP/results/{filename_prefix}_iteration_{iteration}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"📸 Image sauvegardée: {filename}")


def plot_policy(policy_all, V):
    """
    Display and save the optimal policy with all optimal actions
    (Uses the policy with all optimal actions for the final display)
    """
    print("AFFICHAGE DE LA POLITIQUE OPTIMALE (TOUTES LES ACTIONS OPTIMALES)")
    print("=" * 60)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    fig.patch.set_facecolor('#F8F9FA')
    
    arrows = {0: '↑', 1: '↓', 2: '→', 3: '←', -1: '★'}
    arrow_colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1', 3: '#96CEB4', -1: '#FFEAA7'}
    action_names = {0: 'HAUT', 1: 'BAS', 2: 'DROITE', 3: 'GAUCHE', -1: 'TERMINAL'}
    
    for x in range(4):
        for y in range(4):
            y_display = 3 - y
            
            if policy_all[x, y] == [-1]:  # État terminal
                color = "#92BD98"
                border_color = "#1F5024"
                border_width = 3
            else:
                color = "#E3F2FD"
                border_color = "#1C4B6F"
                border_width = 2
            
            rect = plt.Rectangle((x - 0.4, y_display - 0.4), 0.8, 0.8,
                               facecolor=color, edgecolor=border_color, linewidth=2, alpha=0.9)
            ax.add_patch(rect)
            
            # Afficher toutes les actions optimales
            if policy_all[x, y] == [-1]:
                # État terminal
                ax.text(x, y_display, arrows[-1], ha='center', va='center', 
                       fontsize=24, fontweight='bold', color=arrow_colors[-1])
            else:
                # Afficher toutes les actions optimales côte à côte
                optimal_actions = policy_all[x, y]
                if len(optimal_actions) == 1:
                    # Une seule action optimale
                    arrow_text = arrows[optimal_actions[0]]
                    arrow_color = arrow_colors[optimal_actions[0]]
                    ax.text(x, y_display, arrow_text, ha='center', va='center', 
                           fontsize=20, fontweight='bold', color=arrow_color)
                else:
                    # Plusieurs actions optimales - les afficher côte à côte
                    arrow_text = ' '.join([arrows[action] for action in optimal_actions])
                    ax.text(x, y_display, arrow_text, ha='center', va='center', 
                           fontsize=16, fontweight='bold', color='#2C3E50')
            
            # Afficher aussi la valeur V
            ax.text(x, y_display - 0.25, f"V={V[x,y]:.2f}", ha='center', va='center', 
                   fontsize=8, color='#666666')

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_title("POLITIQUE OPTIMALE - POLICY ITERATION\n(Avec toutes les actions optimales possibles)", 
                fontsize=16, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    
    # Légende
    legend_text = "LÉGENDE:\n"
    for action, name in action_names.items():
        if action != -1:
            legend_text += f"{arrows[action]} = {name}\n"
    legend_text += "★ = État terminal\n"
    legend_text += "→ ↑ = Actions multiples optimales"
    
    ax.text(3.8, 2, legend_text, fontsize=10, va='top', 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='white', edgecolor='gray'))
    
    plt.tight_layout()
    
    # Sauvegarder l'image de la politique optimale
    os.makedirs("02_Level1_DP/results", exist_ok=True)
    policy_filename = "02_Level1_DP/results/optimal_policy_all_actions.png"
    plt.savefig(policy_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📸 Politique optimale (toutes actions) sauvegardée: {policy_filename}")
    
    plt.show()


if __name__ == "__main__":
    # Créer le dossier results s'il n'existe pas
    os.makedirs("results", exist_ok=True)
    
    env = PI()
    env.reset()

    gamma = 0.9
    V, policy = policy_iteration(env, gamma=gamma)

    # EXTRAIRE TOUTES LES ACTIONS OPTIMALES seulement pour l'affichage final
    policy_all_optimal = extract_all_optimal_actions(env, V, gamma)
    
    plot_policy(policy_all_optimal, V)
    
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    print("Algorithme Policy Iteration terminé avec succès!")
    print(f"Politique optimale trouvée pour γ={gamma}")
    print("="*60)
    
    # Afficher toutes les actions optimales dans le terminal
    print("\nPOLITIQUE OPTIMALE DÉTAILLÉE (TOUTES LES ACTIONS OPTIMALES):")
    action_names = ["HAUT", "BAS", "DROITE", "GAUCHE"]
    for x in range(4):
        for y in range(4):
            if policy_all_optimal[x, y] == [-1]:
                print(f"  État ({x},{y}): ÉTAT TERMINAL")
            else:
                optimal_actions = [action_names[a] for a in policy_all_optimal[x, y]]
                print(f"  État ({x},{y}): {optimal_actions}")
    
    print("\nPAUSE - Appuyez sur Entrée pour fermer...")
    input()
    
    env.close()
    print("Programme terminé!")