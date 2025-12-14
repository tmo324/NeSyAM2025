#!/usr/bin/env python3
"""
Generate comparison plots for LunarLander results
"""
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


def main():
    print("loading results...")
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    with open(os.path.join(weights_dir, 'ppo_results.pkl'), 'rb') as f:
        ppo_results = pickle.load(f)

    with open(os.path.join(weights_dir, 'viper_results.pkl'), 'rb') as f:
        viper_results = pickle.load(f)

    with open(os.path.join(weights_dir, 'camram_results.pkl'), 'rb') as f:
        camram_results = pickle.load(f)

    with open(os.path.join(weights_dir, 'hybrid_results.pkl'), 'rb') as f:
        hybrid_results = pickle.load(f)

    print("loaded")

    # Prepare data
    results = {
        'PPO\n(Baseline)': ppo_results,
        'VIPER\n(DT)': viper_results,
        'CAM-RAM\n(Memory)': camram_results,
        'Hybrid\n(VIPER+Mem)': hybrid_results
    }

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: All methods comparison
    methods = list(results.keys())
    means = [results[m]['mean_reward'] for m in methods]
    stds = [results[m]['std_reward'] for m in methods]

    x_pos = np.arange(len(methods))
    colors = ['#3498db', '#e74c3c', '#95a5a6', '#2ecc71']
    bars = ax1.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7,
                   color=colors, edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax1.set_title('LunarLander-v3: Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        y_pos = height + std + 20 if height > 0 else height - std - 20
        ax1.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{mean:.1f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=10)

    # Plot 2: Progression view
    progression_methods = ['PPO\n(Baseline)', 'VIPER\n(DT)', 'Hybrid\n(VIPER+Mem)']
    progression_means = [ppo_results['mean_reward'], viper_results['mean_reward'],
                         hybrid_results['mean_reward']]
    progression_stds = [ppo_results['std_reward'], viper_results['std_reward'],
                        hybrid_results['std_reward']]

    x_pos2 = np.arange(len(progression_methods))
    colors2 = ['#3498db', '#e74c3c', '#2ecc71']
    bars2 = ax2.bar(x_pos2, progression_means, yerr=progression_stds, capsize=5,
                    alpha=0.7, color=colors2, edgecolor='black', linewidth=1.5)

    ax2.set_xlabel('Evolution of Approach', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Reward', fontsize=12, fontweight='bold')
    ax2.set_title('PPO ≈ VIPER < VIPER+Memory', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(progression_methods, fontsize=10)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels and improvement arrows
    for i, (bar, mean, std) in enumerate(zip(bars2, progression_means, progression_stds)):
        height = bar.get_height()
        y_pos = height + std + 20 if height > 0 else height - std - 20
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{mean:.1f}',
                ha='center', va='bottom' if height > 0 else 'top',
                fontweight='bold', fontsize=11)

        # Add improvement annotations
        if i < len(bars2) - 1:
            next_mean = progression_means[i + 1]
            improvement = next_mean - mean
            if improvement > 0:
                arrow_text = f'+{improvement:.1f}'
                arrow_color = 'green'
            else:
                arrow_text = f'{improvement:.1f}'
                arrow_color = 'red'

            ax2.annotate(arrow_text,
                        xy=(x_pos2[i] + 0.5, (mean + next_mean) / 2),
                        fontsize=9, fontweight='bold', color=arrow_color,
                        ha='center')

    plt.tight_layout()

    # Save plots
    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    plot_path = os.path.join(plots_dir, 'lunarlander_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')

    print(f"saved to {plot_path}")


if __name__ == "__main__":
    main()
