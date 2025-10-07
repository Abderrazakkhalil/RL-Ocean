import numpy as np
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
from PI_env import PI
import seaborn as sns
from gymnasium import spaces

def policy_evaluation_async(env, policy, V, gamma=0.9, theta=1e-3, max_iterations=1000):
    """
    Asynchronous policy evaluation (in-place updates)
    """
    start_time = time.time()
    iterations = 0
    
    for i in range(max_iterations):
        delta = 0
        
        for x in range(env.grid_width):
            for y in range(env.grid_height):
                state = np.array([x, y])
                
                if tuple(state) in env.terminal_states:
                    continue
                
                action = policy[x, y]
                next_state, reward, terminated, _, _ = env.step_state(state, action)
                nx, ny = next_state
                
                old_value = V[x, y]
                new_value = reward + gamma * V[nx, ny]
                V[x, y] = new_value
                delta = max(delta, abs(new_value - old_value))
        
        iterations = i + 1
        if delta < theta:
            break
    
    eval_time = time.time() - start_time
    return V, iterations, eval_time

def policy_evaluation_sync(env, policy, V, gamma=0.9, theta=1e-3, max_iterations=1000):
    """
    Synchronous policy evaluation (out-of-place updates)
    """
    start_time = time.time()
    iterations = 0
    
    for i in range(max_iterations):
        delta = 0
        V_new = V.copy()
        
        for x in range(env.grid_width):
            for y in range(env.grid_height):
                state = np.array([x, y])
                
                if tuple(state) in env.terminal_states:
                    continue
                
                action = policy[x, y]
                next_state, reward, terminated, _, _ = env.step_state(state, action)
                nx, ny = next_state
                
                new_value = reward + gamma * V[nx, ny]
                delta = max(delta, abs(new_value - V_new[x, y]))
                V_new[x, y] = new_value
        
        V = V_new
        iterations = i + 1
        if delta < theta:
            break
    
    eval_time = time.time() - start_time
    return V, iterations, eval_time

def extract_policy_standard(env, V, gamma=0.9):
    """
    Standard policy extraction (argmax)
    """
    start_time = time.time()
    
    policy = np.zeros((env.grid_width, env.grid_height), dtype=int)
    
    for x in range(env.grid_width):
        for y in range(env.grid_height):
            state = np.array([x, y])
            if tuple(state) in env.terminal_states:
                policy[x, y] = -1
                continue
                
            values = []
            
            for action in range(4):
                next_state, reward, _, _, _ = env.step_state(state, action)
                nx, ny = next_state
                action_value = reward + gamma * V[nx, ny]
                values.append(action_value)
            
            best_action = np.argmax(values)
            policy[x, y] = best_action
    
    extract_time = time.time() - start_time
    return policy, extract_time

def policy_iteration_variant(env, gamma=0.9, theta=1e-3, max_iterations=50, 
                           eval_method='sync', policy_improvement='standard'):
    """
    Policy Iteration with different variants
    """
    print(f"  Variante: γ={gamma}, θ={theta}, {eval_method}, {policy_improvement}")
    
    # Initialization
    V = np.zeros((env.grid_width, env.grid_height))
    policy = np.random.randint(0, 4, (env.grid_width, env.grid_height))
    
    # Mark terminal states
    for x in range(env.grid_width):
        for y in range(env.grid_height):
            if (x, y) in env.terminal_states:
                policy[x, y] = -1
    
    total_start_time = time.time()
    iteration_data = []
    
    for iteration in range(max_iterations):
        iter_start_time = time.time()
        
        # Phase 1: Evaluation (synchronous or asynchronous)
        if eval_method == 'async':
            V, eval_iterations, eval_time = policy_evaluation_async(env, policy, V.copy(), gamma, theta)
        else:  # sync
            V, eval_iterations, eval_time = policy_evaluation_sync(env, policy, V.copy(), gamma, theta)
        
        # Phase 2: Improvement
        if policy_improvement == 'standard':
            new_policy, extract_time = extract_policy_standard(env, V, gamma)
        
        iter_time = time.time() - iter_start_time
        
        # Convergence metrics
        policy_changes = np.sum(policy != new_policy)
        max_value_change = np.max(np.abs(V - (np.zeros_like(V) if iteration == 0 else iteration_data[-1]['V'])))
        
        iteration_data.append({
            'iteration': iteration + 1,
            'eval_iterations': eval_iterations,
            'eval_time': eval_time,
            'extract_time': extract_time,
            'iter_time': iter_time,
            'policy_changes': policy_changes,
            'max_value_change': max_value_change,
            'V': V.copy(),
            'policy': new_policy.copy()
        })
        
        # Stopping condition
        if policy_changes == 0:
            print(f"    ✓ Convergence après {iteration + 1} itérations")
            break
            
        policy = new_policy
    
    total_time = time.time() - total_start_time
    
    performance = {
        'gamma': gamma,
        'theta': theta,
        'eval_method': eval_method,
        'policy_improvement': policy_improvement,
        'total_iterations': len(iteration_data),
        'total_time': total_time,
        'avg_iter_time': total_time / len(iteration_data) if len(iteration_data) > 0 else 0,
        'avg_eval_time': np.mean([data['eval_time'] for data in iteration_data]),
        'avg_extract_time': np.mean([data['extract_time'] for data in iteration_data]),
        'total_eval_iterations': sum([data['eval_iterations'] for data in iteration_data]),
        'converged': len(iteration_data) < max_iterations,
        'final_max_V': np.max(V),
        'iteration_data': iteration_data
    }
    
    return performance

def create_16x16_environment():
    """
    Create a standard 16x16 environment
    """
    env = PI()
    env.grid_width = 16
    env.grid_height = 16
    env.terminal_states = [(0, 0), (15, 15)]
    env.observation_space = spaces.MultiDiscrete([16, 16])
    env.reset()
    return env

def run_gamma_comparison(env, thetas=[1e-3]):
    """
    Compare different gamma values
    """
    print("\n🔍 COMPARAISON DES GAMMA")
    print("=" * 50)
    
    gammas = [0.5,0.6, 0.7, 0.9, 0.95, 0.99, 0.999]
    results = []
    
    for gamma in gammas:
        for theta in thetas:
            print(f"  Test: γ={gamma}, θ={theta}")
            performance = policy_iteration_variant(
                env, gamma=gamma, theta=theta, 
                eval_method='sync', policy_improvement='standard'
            )
            
            results.append(performance)
    
    return results

def run_theta_comparison(env, gamma=0.9):
    """
    Compare different theta values (convergence criterion)
    """
    print("\n🔍 COMPARAISON DES THETA")
    print("=" * 50)
    
    thetas = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    results = []
    
    for theta in thetas:
        print(f"  Test: θ={theta}")
        performance = policy_iteration_variant(
            env, gamma=gamma, theta=theta, 
            eval_method='sync', policy_improvement='standard'
        )
        
        results.append(performance)
    
    return results

def run_eval_method_comparison(env, gamma=0.9, theta=1e-3):
    """
    Compare evaluation methods (synchronous vs asynchronous)
    """
    print("\n🔍 COMPARAISON DES MÉTHODES D'ÉVALUATION")
    print("=" * 50)
    
    eval_methods = ['sync', 'async']
    results = []
    
    for eval_method in eval_methods:
        print(f"  Test: {eval_method}")
        performance = policy_iteration_variant(
            env, gamma=gamma, theta=theta, 
            eval_method=eval_method, policy_improvement='standard'
        )
        
        results.append(performance)
    
    return results

def run_comprehensive_comparison():
    """
    Run a comprehensive comparison of all parameters
    """
    print("🚀 ANALYSE COMPLÈTE DES PARAMÈTRES - GRILLE 16x16")
    print("=" * 70)
    
    env = create_16x16_environment()
    all_results = []
    
    # Test 1: Comparaison Gamma
    gamma_results = run_gamma_comparison(env)
    all_results.extend(gamma_results)
    
    # Test 2: Comparaison Theta
    theta_results = run_theta_comparison(env)
    all_results.extend(theta_results)
    
    # Test 3: Comparaison méthodes d'évaluation
    eval_results = run_eval_method_comparison(env)
    all_results.extend(eval_results)
    
    return all_results

def plot_gamma_comparison(results):
    """
    Plot for gamma comparison
    """
    gamma_results = [r for r in results if 'theta' in r and r['theta'] == 1e-3 and r['eval_method'] == 'sync']
    
    if not gamma_results:
        return
    
    gammas = [r['gamma'] for r in gamma_results]
    times = [r['total_time'] for r in gamma_results]
    iterations = [r['total_iterations'] for r in gamma_results]
    eval_iterations = [r['total_eval_iterations'] for r in gamma_results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Total time vs gamma
    ax1.plot(gammas, times, 'o-', linewidth=3, markersize=8, color='#FF6B6B')
    ax1.set_title('Temps Total vs Gamma', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Gamma (γ)')
    ax1.set_ylabel('Temps Total (s)')
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    
    # Plot 2: Policy iterations vs gamma
    ax2.bar(range(len(gammas)), iterations, color='#4ECDC4', alpha=0.7)
    ax2.set_title('Itérations Policy vs Gamma', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Gamma (γ)')
    ax2.set_ylabel('Itérations Policy')
    ax2.set_xticks(range(len(gammas)))
    ax2.set_xticklabels([f'{g:.3f}' for g in gammas])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Evaluation iterations vs gamma
    ax3.plot(gammas, eval_iterations, 's-', linewidth=3, markersize=8, color='#45B7D1')
    ax3.set_title('Itérations Évaluation vs Gamma', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Gamma (γ)')
    ax3.set_ylabel('Itérations Évaluation Total')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log')
    
    # Plot 4: Final max value vs gamma
    final_values = [r['final_max_V'] for r in gamma_results]
    ax4.plot(gammas, final_values, '^-', linewidth=3, markersize=8, color='#96CEB4')
    ax4.set_title('Valeur Maximale Finale vs Gamma', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Gamma (γ)')
    ax4.set_ylabel('Valeur Maximale V(s)')
    ax4.grid(True, alpha=0.3)
    ax4.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('02_Level1_DP/results/gamma_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_theta_comparison(results):
    """
    Plot for theta comparison
    """
    theta_results = [r for r in results if 'gamma' in r and r['gamma'] == 0.9 and r['eval_method'] == 'sync']
    
    if not theta_results:
        return
    
    thetas = [r['theta'] for r in theta_results]
    times = [r['total_time'] for r in theta_results]
    iterations = [r['total_iterations'] for r in theta_results]
    eval_iterations = [r['total_eval_iterations'] for r in theta_results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Total time vs theta
    ax1.semilogx(thetas, times, 'o-', linewidth=3, markersize=8, color='#FF6B6B')
    ax1.set_title('Temps Total vs Theta', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Theta (θ) - Critère de Convergence')
    ax1.set_ylabel('Temps Total (s)')
    ax1.grid(True, alpha=0.3)
    ax1.invert_xaxis()
    
    # Plot 2: Policy iterations vs theta
    ax2.bar(range(len(thetas)), iterations, color='#4ECDC4', alpha=0.7)
    ax2.set_title('Itérations Policy vs Theta', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Theta (θ)')
    ax2.set_ylabel('Itérations Policy')
    ax2.set_xticks(range(len(thetas)))
    ax2.set_xticklabels([f'{t:.0e}' for t in thetas])
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Evaluation iterations vs theta
    ax3.semilogx(thetas, eval_iterations, 's-', linewidth=3, markersize=8, color='#45B7D1')
    ax3.set_title('Itérations Évaluation vs Theta', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Theta (θ)')
    ax3.set_ylabel('Itérations Évaluation Total')
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()
    
    # Plot 4: Accuracy vs time
    ax4.plot(times, thetas, '^-', linewidth=3, markersize=8, color='#96CEB4')
    ax4.set_title('Précision vs Temps de Calcul', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Temps Total (s)')
    ax4.set_ylabel('Theta (θ) - Précision')
    ax4.grid(True, alpha=0.3)
    ax4.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('02_Level1_DP/results/theta_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_eval_method_comparison(results):
    """
    Plot for evaluation methods comparison
    """
    eval_results = [r for r in results if 'gamma' in r and r['gamma'] == 0.9 and r['theta'] == 1e-3]
    
    if not eval_results:
        return
    
    methods = [r['eval_method'] for r in eval_results]
    times = [r['total_time'] for r in eval_results]
    iterations = [r['total_iterations'] for r in eval_results]
    avg_eval_times = [r['avg_eval_time'] for r in eval_results]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Total time
    bars1 = ax1.bar(methods, times, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    ax1.set_title('Temps Total par Méthode', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Temps Total (s)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Iterations
    bars2 = ax2.bar(methods, iterations, color=['#FF6B6B', '#4ECDC4'], alpha=0.7)
    ax2.set_title('Itérations Policy par Méthode', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Itérations')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Average evaluation time
    bars3 = ax3.bar(methods, avg_eval_times, color=['#45B7D1', '#96CEB4'], alpha=0.7)
    ax3.set_title('Temps d\'Évaluation Moyen', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Temps (s)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Speedup
    if len(times) == 2:
        speedup = times[0] / times[1]  # sync vs async
        ax4.bar(['Speedup'], [speedup], color='#FDCB6E', alpha=0.7)
        ax4.set_title(f'Speedup Async/Sync: {speedup:.2f}x', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Facteur d\'Accélération')
        ax4.grid(True, alpha=0.3)
    
    # Add values on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax = bar.axes
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('02_Level1_DP/results/eval_method_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_parameter_report(results):
    """
    Generate a detailed report of parameter comparisons
    """
    print("\n📊 RAPPORT DÉTAILLÉ DES PARAMÈTRES")
    print("=" * 80)
    
    # Prepare data for the DataFrame
    report_data = []
    
    for result in results:
        report_data.append({
            'Gamma': result['gamma'],
            'Theta': result['theta'],
            'Méthode Éval': result['eval_method'],
            'Itérations Policy': result['total_iterations'],
            'Itérations Éval Total': result['total_eval_iterations'],
            'Temps Total (s)': f"{result['total_time']:.2f}",
            'Temps Moyen Itération (s)': f"{result['avg_iter_time']:.2f}",
            'Temps Éval Moyen (s)': f"{result['avg_eval_time']:.4f}",
            'Temps Extract Moyen (s)': f"{result['avg_extract_time']:.4f}",
            'Valeur Max Finale': f"{result['final_max_V']:.3f}",
            'Convergence': '✓' if result['converged'] else '✗'
        })
    
    # Create and display the DataFrame
    df = pd.DataFrame(report_data)
    print(df.to_string(index=False))
    
    # Save the report
    df.to_csv('02_Level1_DP/results/parameter_comparison_report.csv', index=False)
    print(f"\n💾 Rapport sauvegardé: 02_Level1_DP/results/parameter_comparison_report.csv")
    
    # Best parameters analysis
    print("\n🏆 MEILLEURS PARAMÈTRES")
    print("-" * 40)
    
    # Fastest
    fastest = min(results, key=lambda x: x['total_time'])
    print(f"Plus rapide: γ={fastest['gamma']}, θ={fastest['theta']}, {fastest['eval_method']}")
    print(f"  Temps: {fastest['total_time']:.2f}s, Itérations: {fastest['total_iterations']}")
    
    # Fewest iterations
    least_iterations = min(results, key=lambda x: x['total_iterations'])
    print(f"Moins d'itérations: γ={least_iterations['gamma']}, θ={least_iterations['theta']}, {least_iterations['eval_method']}")
    print(f"  Itérations: {least_iterations['total_iterations']}, Temps: {least_iterations['total_time']:.2f}s")
    
    # Best convergence (highest precision)
    most_precise = min(results, key=lambda x: x['theta'])
    print(f"Plus précis: θ={most_precise['theta']}, γ={most_precise['gamma']}, {most_precise['eval_method']}")
    print(f"  Précision: {most_precise['theta']:.0e}, Temps: {most_precise['total_time']:.2f}s")

def main():
    """
    Main function
    """
    # Create the results folder
    os.makedirs("results", exist_ok=True)
    
    print("🎯 ANALYSE DES PARAMÈTRES - POLICY ITERATION")
    print("Grille: 16x16, États terminaux: (0,0) et (15,15)")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        # Run the comprehensive comparison
        results = run_comprehensive_comparison()
        
        # Generate plots
        plot_gamma_comparison(results)
        plot_theta_comparison(results)
        plot_eval_method_comparison(results)
        
        # Generate the report
        generate_parameter_report(results)
        
        total_time = time.time() - start_time
        print(f"\n✅ Analyse des paramètres terminée en {total_time:.2f} secondes")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()