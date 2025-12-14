#!/usr/bin/env python3
"""
Clean VIPER tree visualization
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.tree import _tree

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from symbolic.viper_simple import VIPERAgent


def plot_simple_tree(tree, feature_names, action_names, max_depth=3):
    tree_ = tree.tree_
    feature_name = tree_.feature
    threshold = tree_.threshold

    # Figure setup
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # Colors for actions
    action_colors = {
        0: '#e3f2fd',  # Do Nothing - light blue
        1: '#fff3e0',  # Fire Left - light orange
        2: '#f3e5f5',  # Fire Main - light purple
        3: '#e8f5e9'   # Fire Right - light green
    }

    def get_node_action(node_id):
        """Get the dominant action for a node"""
        value = tree_.value[node_id][0]
        return int(np.argmax(value))

    def draw_node(node_id, x, y, width, depth, parent_x=None, parent_y=None):
        """Recursively draw tree nodes"""

        if depth > max_depth:
            return

        # Determine if leaf node
        # It's a leaf if: (1) it's a natural leaf, OR (2) we're at max_depth
        is_leaf = (tree_.feature[node_id] == _tree.TREE_UNDEFINED) or (depth == max_depth)

        if is_leaf:
            # Leaf node - show action
            action = get_node_action(node_id)
            label = action_names[action]
            box_color = action_colors[action]
            box_style = mpatches.FancyBboxPatch(
                (x - 0.04, y - 0.015), 0.08, 0.03,
                boxstyle="round,pad=0.005",
                facecolor=box_color,
                edgecolor='black',
                linewidth=2
            )
            ax.add_patch(box_style)
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=11, fontweight='bold')
        else:
            # Internal node - show condition
            feature = feature_names[feature_name[node_id]]
            thresh = threshold[node_id]
            label = f"{feature} ≤ {thresh:.2f}"

            box_style = mpatches.FancyBboxPatch(
                (x - 0.05, y - 0.015), 0.10, 0.03,
                boxstyle="round,pad=0.005",
                facecolor='#bbdefb',
                edgecolor='black',
                linewidth=1.5
            )
            ax.add_patch(box_style)
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=10, fontweight='bold')

            # Draw children
            left_child = tree_.children_left[node_id]
            right_child = tree_.children_right[node_id]

            new_width = width / 2
            new_y = y - 0.15

            # Left child (True branch)
            left_x = x - width / 4
            if left_child != _tree.TREE_UNDEFINED:
                # Draw line with "True" label
                ax.plot([x, left_x], [y - 0.015, new_y + 0.015],
                       'k-', linewidth=1.5, alpha=0.6)
                ax.text((x + left_x) / 2 - 0.01, (y + new_y) / 2, 'T',
                       fontsize=9, color='green', fontweight='bold',
                       bbox=dict(boxstyle='circle', facecolor='white',
                                edgecolor='green', linewidth=1))
                draw_node(left_child, left_x, new_y, new_width, depth + 1, x, y)

            # Right child (False branch)
            right_x = x + width / 4
            if right_child != _tree.TREE_UNDEFINED:
                # Draw line with "False" label
                ax.plot([x, right_x], [y - 0.015, new_y + 0.015],
                       'k-', linewidth=1.5, alpha=0.6)
                ax.text((x + right_x) / 2 + 0.01, (y + new_y) / 2, 'F',
                       fontsize=9, color='red', fontweight='bold',
                       bbox=dict(boxstyle='circle', facecolor='white',
                                edgecolor='red', linewidth=1))
                draw_node(right_child, right_x, new_y, new_width, depth + 1, x, y)

    # Draw from root
    draw_node(0, 0.5, 0.95, 0.8, 0)

    # Add legend
    legend_elements = [
        mpatches.Patch(facecolor=action_colors[0], edgecolor='black', label='Do Nothing'),
        mpatches.Patch(facecolor=action_colors[1], edgecolor='black', label='Fire Left'),
        mpatches.Patch(facecolor=action_colors[2], edgecolor='black', label='Fire Main'),
        mpatches.Patch(facecolor=action_colors[3], edgecolor='black', label='Fire Right')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)

    # Title
    plt.suptitle(f'VIPER Decision Tree (Simplified - Top {max_depth} Levels)',
                fontsize=16, fontweight='bold', y=0.98)

    # Add note
    ax.text(0.5, 0.02,
           'Blue boxes = conditions | Colored boxes = actions | T/F = True/False branch',
           ha='center', fontsize=10, style='italic', color='gray')

    return fig, ax


def main():
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    viper_path = os.path.join(weights_dir, 'viper_agent.pkl')

    print(f"loading {viper_path}")
    viper_agent = VIPERAgent.load(viper_path)

    # Feature and action names
    feature_names = [
        'x_pos', 'y_pos', 'x_vel', 'y_vel',
        'angle', 'angular_vel', 'left_leg', 'right_leg'
    ]
    action_names = ['Do Nothing', 'Fire Left', 'Fire Main', 'Fire Right']

    print("creating clean tree (depth=3)...")
    fig, ax = plot_simple_tree(viper_agent.tree, feature_names, action_names, max_depth=3)

    plots_dir = os.path.join(os.path.dirname(__file__), 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    output_path = os.path.join(plots_dir, 'viper_tree_clean.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"saved to {output_path}")

    print("creating ultra-simple tree (depth=2)...")
    fig2, ax2 = plot_simple_tree(viper_agent.tree, feature_names, action_names, max_depth=2)
    plt.suptitle('VIPER Decision Tree (Ultra-Simple - Top 2 Levels)',
                fontsize=16, fontweight='bold', y=0.98)

    output_path2 = os.path.join(plots_dir, 'viper_tree_ultra_simple.png')
    plt.savefig(output_path2, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"saved to {output_path2}")
    print("done")


if __name__ == "__main__":
    main()
