import numpy as np
import matplotlib.pyplot as plt
import time
import os
from gymnasium  import spaces
import pandas as pd
from PI_env import PI

def policy_evaluation_perf(env, policy, V, gamma=0.9, theta=1e-3, max_iterations=1000):
    """
    Policy evaluation with time measurement
    """
    start_time = time.time()
    iterations = 0
    
    for i in range(max_iterations):
        delta = 0
        V_prev = V.copy()
        
        for x in range(env.grid_width):
            for y in range(env.grid_height):
                state = np.array([x, y])
                
        # Terminal states - no update
                if tuple(state) in env.terminal_states:
                    continue
                
                # Get the action according to the current policy
                action = policy[x, y]
                next_state, reward, terminated, _, _ = env.step_state(state, action)
                nx, ny = next_state
                
                # Value update
                new_value = reward + gamma * V_prev[nx, ny]
                delta = max(delta, abs(new_value - V[x, y]))
                V[x, y] = new_value
        
        iterations = i + 1
        if delta < theta:
            break
    
    eval_time = time.time() - start_time
    return V, iterations, eval_time


def extract_policy_perf(env, V, gamma=0.9):
    """
    Policy extraction with time measurement
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


def policy_iteration_perf(env, gamma=0.9, theta=1e-3, max_iterations=50):
    """
    Policy Iteration with detailed performance measurements
    """
    print(f"  Démarrage Policy Iteration pour grille {env.grid_width}x{env.grid_height}")
    
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
        
        # Phase 1: Evaluation
        V, eval_iterations, eval_time = policy_evaluation_perf(env, policy, V.copy(), gamma, theta)
        
        # Phase 2: Improvement
        new_policy, extract_time = extract_policy_perf(env, V, gamma)
        
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
            print(f"    Convergence après {iteration + 1} itérations")
            break
            
        policy = new_policy
    
    total_time = time.time() - total_start_time
    
    # Performance summary
    performance = {
        'grid_size': f"{env.grid_width}x{env.grid_height}",
        'total_iterations': len(iteration_data),
        'total_time': total_time,
        'avg_iter_time': total_time / len(iteration_data) if len(iteration_data) > 0 else 0,
        'converged': len(iteration_data) < max_iterations,
        'iteration_data': iteration_data,
        'final_V': V,
        'final_policy': policy
    }
    
    return performance


def create_test_environments():
    """
    Create test environments for different grid sizes
    """
    environments = {}
    
    # 4x4 grid
    environments['4x4'] = PI()
    
    # 8x8 grid
    env_8x8 = PI()
    env_8x8.grid_width = 8
    env_8x8.grid_height = 8
    env_8x8.terminal_states = [(0, 0), (7, 7)]
    env_8x8.observation_space = spaces.MultiDiscrete([8, 8])
    environments['8x8'] = env_8x8
    
    # 16x16 grid
    env_16x16 = PI()
    env_16x16.grid_width = 16
    env_16x16.grid_height = 16
    env_16x16.terminal_states = [(0, 0), (15, 15)]
    env_16x16.observation_space = spaces.MultiDiscrete([16, 16])
    environments['16x16'] = env_16x16
    
    # 32x32 grid
    env_32x32 = PI()
    env_32x32.grid_width = 32
    env_32x32.grid_height = 32
    env_32x32.terminal_states = [(0, 0), (31, 31)]
    env_32x32.observation_space = spaces.MultiDiscrete([32, 32])
    environments['32x32'] = env_32x32
    
    return environments


def run_performance_comparison(gamma=0.9, theta=1e-3):
    """
    Run performance comparison for all grid sizes
    """
    print("COMPARAISON DES PERFORMANCES - POLICY ITERATION")
    print("=" * 60)
    
    # Create test environments
    environments = create_test_environments()
    results = {}
    
    for size_name, env in environments.items():
        print(f"\n🔍 Test en cours: Grille {size_name}")
        print("-" * 40)
        
        env.reset()
        performance = policy_iteration_perf(env, gamma, theta)
        results[size_name] = performance
        
        # Display results for this size
        print(f"  ✓ Terminé: {performance['total_iterations']} itérations")
        print(f"  ✓ Temps total: {performance['total_time']:.2f} secondes")
        print(f"  ✓ Convergence: {'OUI' if performance['converged'] else 'NON'}")
        print(f"  ✓ Temps moyen par itération: {performance['avg_iter_time']:.2f}s")
    
    return results


def plot_performance_results(results):
    """
    Create performance plots
    """
    print("\n📊 Génération des graphiques de performance...")
    
    # Prepare data
    sizes = list(results.keys())
    total_times = [results[size]['total_time'] for size in sizes]
    total_iterations = [results[size]['total_iterations'] for size in sizes]
    avg_iter_times = [results[size]['avg_iter_time'] for size in sizes]
    
    # Plot 1: Total time vs grid size
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(sizes, total_times, 'o-', linewidth=2, markersize=8, color='#FF6B6B')
    plt.title('Temps Total d\'Exécution', fontsize=14, fontweight='bold')
    plt.xlabel('Taille de Grille')
    plt.ylabel('Temps (secondes)')
    plt.grid(True, alpha=0.3)
    
    # Annotate points
    for i, (size, time_val) in enumerate(zip(sizes, total_times)):
        plt.annotate(f'{time_val:.1f}s', (i, time_val), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontweight='bold')
    
    # Plot 2: Number of iterations
    plt.subplot(2, 2, 2)
    plt.bar(sizes, total_iterations, color='#4ECDC4', alpha=0.7)
    plt.title('Nombre d\'Itérations', fontsize=14, fontweight='bold')
    plt.xlabel('Taille de Grille')
    plt.ylabel('Itérations')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Average time per iteration
    plt.subplot(2, 2, 3)
    plt.plot(sizes, avg_iter_times, 's-', linewidth=2, markersize=8, color='#45B7D1')
    plt.title('Temps Moyen par Itération', fontsize=14, fontweight='bold')
    plt.xlabel('Taille de Grille')
    plt.ylabel('Temps (secondes)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Computational complexity
    plt.subplot(2, 2, 4)
    grid_sizes_num = [int(size.split('x')[0]) for size in sizes]
    theoretical_complexity = [size**4 for size in grid_sizes_num]  # O(n^4) for Policy Iteration
    theoretical_complexity = [tc / theoretical_complexity[0] * total_times[0] for tc in theoretical_complexity]
    
    plt.plot(sizes, total_times, 'o-', linewidth=2, markersize=8, color='#FF6B6B', label='Real Time')
    plt.plot(sizes, theoretical_complexity, '--', linewidth=2, color='#96CEB4', label='Theoretical Complexity O(n⁴)')
    plt.title('Computational Complexity', fontsize=14, fontweight='bold')
    plt.xlabel('Grid Size')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('02_Level1_DP/results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Detailed plot: Convergence evolution
    plt.figure(figsize=(15, 10))
    
    for i, size_name in enumerate(sizes):
        iteration_data = results[size_name]['iteration_data']
        iterations = [data['iteration'] for data in iteration_data]
        policy_changes = [data['policy_changes'] for data in iteration_data]
        value_changes = [data['max_value_change'] for data in iteration_data]
        
        plt.subplot(2, 2, i+1)
        plt.plot(iterations, policy_changes, 'o-', label='Policy Changes', linewidth=2)
        plt.plot(iterations, value_changes, 's-', label='Max Change of V', linewidth=2)
        plt.title(f'Convergence - Grid {size_name}', fontsize=12, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Convergence Measure')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Mark convergence
        if results[size_name]['converged']:
            conv_iter = len(iterations)
            plt.axvline(x=conv_iter, color='red', linestyle='--', alpha=0.7)
            plt.text(conv_iter, max(max(policy_changes), max(value_changes)), 
                    f'Convergence\niteration {conv_iter}', 
                    ha='center', va='top', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('02_Level1_DP/results/convergence_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_performance_report(results):
    """
    Generate a detailed performance report
    """
    print("\n📈 RAPPORT DE PERFORMANCE DÉTAILLÉ")
    print("=" * 80)
    
    report_data = []
    
    for size_name, perf in results.items():
        iteration_data = perf['iteration_data']
        
    # Detailed statistics
        eval_times = [data['eval_time'] for data in iteration_data]
        extract_times = [data['extract_time'] for data in iteration_data]
        iter_times = [data['iter_time'] for data in iteration_data]
        
        report_data.append({
            'Grid Size': size_name,
            'Itérations Total': perf['total_iterations'],
            'Total Time (s)': f"{perf['total_time']:.2f}",
            'Avg Iteration Time (s)': f"{perf['avg_iter_time']:.2f}",
            'Avg Evaluation Time (s)': f"{np.mean(eval_times):.4f}",
            'Avg Extraction Time (s)': f"{np.mean(extract_times):.4f}",
            'Convergence': '✓' if perf['converged'] else '✗',
            'States': f"{int(size_name.split('x')[0]) ** 2}"
        })
    
    # Create a DataFrame for clean display
    df = pd.DataFrame(report_data)
    print(df.to_string(index=False))
    
    # Save the CSV report
    df.to_csv('02_Level1_DP/results/performance_report.csv', index=False)
    print(f"\n💾 Rapport sauvegardé: results/performance_report.csv")
    
    # Complexity analysis
    print("\n🔬 ANALYSE DE COMPLEXITÉ")
    print("-" * 40)
    
    sizes_num = [int(size.split('x')[0]) for size in results.keys()]
    times = [results[size]['total_time'] for size in results.keys()]
    
    for i in range(1, len(sizes_num)):
        size_ratio = sizes_num[i] / sizes_num[i-1]
        time_ratio = times[i] / times[i-1]
        print(f"  {sizes_num[i-1]}x{sizes_num[i-1]} → {sizes_num[i]}x{sizes_num[i]}: "
              f"Size ×{size_ratio:.1f}, Time ×{time_ratio:.1f}")


def main():
    """
    Main function to run the performance comparison
    """
    # Create the results folder
    os.makedirs("results", exist_ok=True)
    
    print("🚀 DÉMARRAGE DE L'ANALYSE DE PERFORMANCE")
    print("Test des grilles: 4x4, 8x8, 16x16, 32x32")
    print("Paramètres: γ=0.9, θ=0.001")
    print("=" * 60)
    
    start_time = time.time()
    
    try:
        # Run the comparison
        results = run_performance_comparison(gamma=0.9, theta=1e-3)
        
        # Generate plots
        plot_performance_results(results)
        
        # Generate the report
        generate_performance_report(results)
        
        total_time = time.time() - start_time
        print(f"\n✅ Analyse terminée en {total_time:.2f} secondes")
        
    except Exception as e:
        print(f"\n❌ Erreur lors de l'analyse: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()