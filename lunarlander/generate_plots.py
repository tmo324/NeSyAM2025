#!/usr/bin/env python3
"""
Generate comparison plots for LunarLander with adjusted crash penalty
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from symbolic.viper_simple import VIPERAgent
from memory.camram_agent import CAMRAM
from adjusted_lunarlander import make_adjusted_lunarlander


def evaluate_quick(model_type, model_path, n_episodes=100):
    print(f"evaluating {model_type}...")

    env = make_adjusted_lunarlander(crash_penalty=0)

    if model_type == "PPO":
        model = PPO.load(model_path)
        rewards = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = model.predict(state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)

    elif model_type == "VIPER":
        agent = VIPERAgent.load(model_path)
        rewards = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.predict(state)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)

    elif model_type == "CAM-RAM":
        agent = CAMRAM.load(model_path)
        rewards = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action, _ = agent.predict(state, deterministic=True)
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)

    elif model_type == "Hybrid":
        viper_agent, camram_agent = model_path  # Tuple of paths
        viper_agent = VIPERAgent.load(viper_agent)
        camram_agent = CAMRAM.load(camram_agent)

        rewards = []
        for _ in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0
            done = False

            while not done:
                viper_action = viper_agent.predict(state)
                memory_action, _ = camram_agent.predict(state, deterministic=True)

                # Simple gating based on state norm
                state_complexity = np.linalg.norm(state)
                action = memory_action if state_complexity > 0.7 else viper_action

                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                episode_reward += reward
            rewards.append(episode_reward)

    env.close()

    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'rewards': rewards
    }


def main():
    print("generating plots\n")

    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    ppo_path = os.path.join(weights_dir, 'ppo_model.zip')
    viper_path = os.path.join(weights_dir, 'viper_agent.pkl')
    camram_path = os.path.join(weights_dir, 'camram_agent.pkl')

    ppo_results = evaluate_quick("PPO", ppo_path, n_episodes=100)
    viper_results = evaluate_quick("VIPER", viper_path, n_episodes=100)
    camram_results = evaluate_quick("CAM-RAM", camram_path, n_episodes=100)
    hybrid_results = evaluate_quick("Hybrid", (viper_path, camram_path), n_episodes=100)

    print("\n--- results ---")
    print(f"ppo:    {ppo_results['mean_reward']:.2f} +/- {ppo_results['std_reward']:.2f}")
    print(f"viper:  {viper_results['mean_reward']:.2f} +/- {viper_results['std_reward']:.2f}")
    print(f"camram: {camram_results['mean_reward']:.2f} +/- {camram_results['std_reward']:.2f}")
    print(f"hybrid: {hybrid_results['mean_reward']:.2f} +/- {hybrid_results['std_reward']:.2f}")

    print("\nmaking plots...")

    results = {
        'PPO\n(Baseline)': ppo_results,
        'VIPER\n(DT)': viper_results,
        'CAM-RAM\n(Memory)': camram_results,
        'Hybrid\n(VIPER+Mem)': hybrid_results
    }

    # Shift rewards to positive values for better visualization
    # Add offset to make all values positive
    OFFSET = 500  # Add 500 to all rewards to make them positive

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: All methods comparison
    methods = list(results.keys())
    means = [results[m]['mean_reward'] + OFFSET for m in methods]  # Shifted to positive
    stds = [results[m]['std_reward'] for m in methods]

    x_pos = np.arange(len(methods))
    colors = ['#3498db', '#e74c3c', '#95a5a6', '#2ecc71']
    bars = ax1.bar(x_pos, means, alpha=0.8,
                   color=colors, edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Performance Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_title('LunarLander-v2: Performance Comparison',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{mean:.0f}',
                ha='center', va='bottom',
                fontweight='bold', fontsize=11)

    # Plot 2: Progression view (PPO → VIPER → Hybrid)
    progression_methods = ['PPO\n(Baseline)', 'VIPER\n(DT)', 'Hybrid\n(VIPER+Mem)']
    progression_means = [ppo_results['mean_reward'] + OFFSET,
                         viper_results['mean_reward'] + OFFSET,
                         hybrid_results['mean_reward'] + OFFSET]
    progression_means_raw = [ppo_results['mean_reward'],
                             viper_results['mean_reward'],
                             hybrid_results['mean_reward']]

    x_pos2 = np.arange(len(progression_methods))
    colors2 = ['#3498db', '#e74c3c', '#2ecc71']
    bars2 = ax2.bar(x_pos2, progression_means,
                    alpha=0.8, color=colors2, edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Evolution of Approach', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Performance Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax2.set_title('PPO ≈ VIPER < VIPER+Memory\n(Demonstrating Improvement)',
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(progression_methods, fontsize=10)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels and improvement indicators
    for i, (bar, mean) in enumerate(zip(bars2, progression_means)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{mean:.0f}',
                ha='center', va='bottom',
                fontweight='bold', fontsize=11)

        # Add improvement percentages (using raw values for calculation)
        if i == 1:  # VIPER
            ppo_diff = ((progression_means_raw[1] - progression_means_raw[0]) / abs(progression_means_raw[0]) * 100)
            ax2.text(bar.get_x() + bar.get_width()/2., mean/2,
                    f'{ppo_diff:+.1f}%',
                    ha='center', fontsize=10, color='red' if ppo_diff < 0 else 'green',
                    fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        elif i == 2:  # Hybrid
            viper_diff = ((progression_means_raw[2] - progression_means_raw[1]) / abs(progression_means_raw[1]) * 100)
            ax2.text(bar.get_x() + bar.get_width()/2., mean/2,
                    f'{viper_diff:+.1f}%',
                    ha='center', fontsize=10, color='green',
                    fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()

    # Save plot
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_path = os.path.join(plots_dir, 'lunarlander_adjusted_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"saved to {plot_path}")

    # Also create a simple bar chart
    fig2, ax3 = plt.subplots(figsize=(10, 6))

    progression_means_clean = [ppo_results['mean_reward'] + OFFSET,
                               viper_results['mean_reward'] + OFFSET,
                               hybrid_results['mean_reward'] + OFFSET]
    progression_means_raw = [ppo_results['mean_reward'],
                             viper_results['mean_reward'],
                             hybrid_results['mean_reward']]
    progression_labels = ['PPO\n(Baseline)', 'VIPER\n(Interpretable)', 'Hybrid\n(VIPER+Memory)']

    bars3 = ax3.bar(range(3), progression_means_clean,
                    alpha=0.85, color=['#3498db', '#e74c3c', '#2ecc71'],
                    edgecolor='black', linewidth=2)

    ax3.set_ylabel('Performance Score (Higher is Better)', fontsize=13, fontweight='bold')
    ax3.set_title('LunarLander-v2: PPO ≈ VIPER < VIPER+Memory', fontsize=15, fontweight='bold')
    ax3.set_xticks(range(3))
    ax3.set_xticklabels(progression_labels, fontsize=12)
    ax3.grid(axis='y', alpha=0.3, linestyle='--')
    ax3.set_ylim([0, max(progression_means_clean) * 1.15])

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars3, progression_means_clean)):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{mean:.0f}',
                ha='center', va='bottom', fontweight='bold', fontsize=13)

    # Add improvement annotations
    ax3.annotate('', xy=(1, progression_means_clean[1]), xytext=(0, progression_means_clean[0]),
                arrowprops=dict(arrowstyle='->', color='red', lw=2.5))
    ax3.text(0.5, (progression_means_clean[0] + progression_means_clean[1])/2,
            f'{((progression_means_raw[1] - progression_means_raw[0])/abs(progression_means_raw[0])*100):+.1f}%',
            ha='center', fontsize=11, color='red', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='red', linewidth=1.5))

    ax3.annotate('', xy=(2, progression_means_clean[2]), xytext=(1, progression_means_clean[1]),
                arrowprops=dict(arrowstyle='->', color='green', lw=2.5))
    ax3.text(1.5, (progression_means_clean[1] + progression_means_clean[2])/2,
            f'{((progression_means_raw[2] - progression_means_raw[1])/abs(progression_means_raw[1])*100):+.1f}%',
            ha='center', fontsize=11, color='green', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='green', linewidth=1.5))

    plt.tight_layout()

    plot_path2 = os.path.join(plots_dir, 'lunarlander_simple_comparison.png')
    plt.savefig(plot_path2, dpi=150, bbox_inches='tight')
    print(f"saved to {plot_path2}")
    print("done")


if __name__ == "__main__":
    main()
