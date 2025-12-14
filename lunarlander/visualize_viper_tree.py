#!/usr/bin/env python3
"""
Visualize VIPER decision tree
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from symbolic.viper_simple import VIPERAgent


def main():
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    viper_path = os.path.join(weights_dir, 'viper_agent.pkl')

    print(f"loading {viper_path}")
    viper_agent = VIPERAgent.load(viper_path)

    # LunarLander state space feature names
    feature_names = [
        'x_pos',           # 0: horizontal position
        'y_pos',           # 1: vertical position
        'x_vel',           # 2: horizontal velocity
        'y_vel',           # 3: vertical velocity
        'angle',           # 4: angle
        'angular_vel',     # 5: angular velocity
        'left_leg',        # 6: left leg contact
        'right_leg'        # 7: right leg contact
    ]

    # Action names
    action_names = ['Do Nothing', 'Fire Left', 'Fire Main', 'Fire Right']

    print("creating full tree...")

    fig, ax = plt.subplots(figsize=(25, 15))
    plot_tree(
        viper_agent.tree,
        feature_names=feature_names,
        class_names=action_names,
        filled=True,
        rounded=True,
        fontsize=8,
        ax=ax,
        proportion=True
    )
    ax.set_title('VIPER Decision Tree (Full)', fontsize=16, fontweight='bold', pad=20)

    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    full_tree_path = os.path.join(plots_dir, 'viper_tree_full.png')
    plt.savefig(full_tree_path, dpi=200, bbox_inches='tight')
    print(f"saved to {full_tree_path}")

    print("creating simplified tree...")

    fig2, ax2 = plt.subplots(figsize=(20, 12))
    plot_tree(
        viper_agent.tree,
        feature_names=feature_names,
        class_names=action_names,
        filled=True,
        rounded=True,
        fontsize=10,
        ax=ax2,
        max_depth=3,  # Show only top 3 levels
        proportion=True
    )
    ax2.set_title('VIPER Decision Tree (Simplified - Top 3 Levels)',
                  fontsize=16, fontweight='bold', pad=20)

    simple_tree_path = os.path.join(plots_dir, 'viper_tree_simplified.png')
    plt.savefig(simple_tree_path, dpi=200, bbox_inches='tight')
    print(f"saved to {simple_tree_path}")

    print("generating text representation...")

    tree_text = export_text(
        viper_agent.tree,
        feature_names=feature_names,
        max_depth=4,  # Limit depth for readability
        show_weights=True
    )

    text_output_path = os.path.join(plots_dir, 'viper_tree_text.txt')
    with open(text_output_path, 'w') as f:
        f.write("VIPER Decision Tree - Text Representation\n")
        f.write("=" * 70 + "\n")
        f.write(f"Tree Depth: {viper_agent.get_depth()}\n")
        f.write(f"Number of Leaves: {viper_agent.get_num_leaves()}\n")
        f.write(f"Training Accuracy: {viper_agent.training_accuracy:.4f}\n")
        f.write("=" * 70 + "\n\n")
        f.write("Action Mapping:\n")
        for i, action in enumerate(action_names):
            f.write(f"  {i}: {action}\n")
        f.write("\n" + "=" * 70 + "\n\n")
        f.write(tree_text)
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("Note: Tree shows top 4 levels. Full tree has depth " +
                f"{viper_agent.get_depth()}\n")

    print(f"saved to {text_output_path}")

    print("analyzing feature importance...")

    feature_importance = viper_agent.tree.feature_importances_

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))

    bars = ax3.barh(feature_names, feature_importance, color=colors, edgecolor='black', linewidth=1.5)
    ax3.set_xlabel('Importance', fontsize=12, fontweight='bold')
    ax3.set_ylabel('State Features', fontsize=12, fontweight='bold')
    ax3.set_title('VIPER Tree: Feature Importance\n(Which states matter most for decisions)',
                  fontsize=14, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)

    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, feature_importance)):
        if importance > 0.01:  # Only show if > 1%
            ax3.text(importance + 0.01, i, f'{importance:.3f}',
                    va='center', fontweight='bold', fontsize=10)

    plt.tight_layout()

    importance_path = os.path.join(plots_dir, 'viper_feature_importance.png')
    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
    print(f"saved to {importance_path}")
    print("done")


if __name__ == "__main__":
    main()
