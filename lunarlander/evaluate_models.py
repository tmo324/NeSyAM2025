#!/usr/bin/env python3
"""
Evaluation script for LunarLander - compares PPO, VIPER, CAM-RAM, and Hybrid
"""
import os
import sys
import pickle
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from symbolic.viper_simple import VIPERAgent
from memory.camram_agent import CAMRAM
from adjusted_lunarlander import make_adjusted_lunarlander


def evaluate_ppo(model_path, env_name, n_episodes=100):
    print("\n--- ppo ---")

    model = PPO.load(model_path)
    env = make_adjusted_lunarlander(crash_penalty=0)

    rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = model.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print(f"ppo: {mean_reward:.2f} +/- {std_reward:.2f}")

    return {'mean_reward': mean_reward, 'std_reward': std_reward}


def evaluate_viper(agent_path, env_name, n_episodes=100):
    print("\n--- viper ---")

    viper_agent = VIPERAgent.load(agent_path)
    env = make_adjusted_lunarlander(crash_penalty=0)

    rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = viper_agent.predict(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print(f"viper: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  depth={viper_agent.get_depth()}, leaves={viper_agent.get_num_leaves()}")

    return {'mean_reward': mean_reward, 'std_reward': std_reward}


def evaluate_camram(agent_path, env_name, n_episodes=100):
    print("\n--- camram ---")

    camram_agent = CAMRAM.load(agent_path)
    env = make_adjusted_lunarlander(crash_penalty=0)

    rewards = []
    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = camram_agent.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    print(f"camram: {mean_reward:.2f} +/- {std_reward:.2f}")

    return {'mean_reward': mean_reward, 'std_reward': std_reward}


def evaluate_hybrid(viper_path, camram_path, env_name, n_episodes=100, confidence_threshold=0.7):
    print("\n--- hybrid ---")

    viper_agent = VIPERAgent.load(viper_path)
    camram_agent = CAMRAM.load(camram_path)
    env = make_adjusted_lunarlander(crash_penalty=0)

    rewards = []
    viper_selections = 0
    memory_selections = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Get predictions from both agents
            viper_action = viper_agent.predict(state)
            memory_action, info = camram_agent.predict(state, deterministic=True)

            # Simple gating based on state norm (proxy for complexity)
            # Use memory for complex states, VIPER for simple ones
            state_complexity = np.linalg.norm(state)

            if state_complexity > confidence_threshold:
                action = memory_action
                memory_selections += 1
            else:
                action = viper_action
                viper_selections += 1

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        rewards.append(episode_reward)

    env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    total_decisions = viper_selections + memory_selections
    viper_pct = (viper_selections / total_decisions * 100) if total_decisions > 0 else 0
    memory_pct = (memory_selections / total_decisions * 100) if total_decisions > 0 else 0

    print(f"hybrid: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  viper: {viper_selections} ({viper_pct:.1f}%), memory: {memory_selections} ({memory_pct:.1f}%)")

    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'viper_percentage': viper_pct,
        'memory_percentage': memory_pct
    }


def main():
    print("lunarlander evaluation")
    print("using adjusted crash penalty (0 instead of -100)\n")

    # Paths
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights')
    ppo_path = os.path.join(weights_dir, 'ppo_model.zip')
    viper_path = os.path.join(weights_dir, 'viper_agent.pkl')
    camram_path = os.path.join(weights_dir, 'camram_agent.pkl')

    env_name = "LunarLander-v2"
    n_episodes = 100

    # Check if weights exist
    if not all(os.path.exists(p) for p in [ppo_path, viper_path, camram_path]):
        print("ERROR: Model weights not found!")
        print(f"Expected weights in: {weights_dir}")
        sys.exit(1)

    # Evaluate all approaches
    ppo_results = evaluate_ppo(ppo_path, env_name, n_episodes)
    viper_results = evaluate_viper(viper_path, env_name, n_episodes)
    camram_results = evaluate_camram(camram_path, env_name, n_episodes)
    hybrid_results = evaluate_hybrid(viper_path, camram_path, env_name, n_episodes)

    # Summary
    print("\n--- results ---")
    print(f"ppo:    {ppo_results['mean_reward']:.2f} +/- {ppo_results['std_reward']:.2f}")
    print(f"viper:  {viper_results['mean_reward']:.2f} +/- {viper_results['std_reward']:.2f}")
    print(f"camram: {camram_results['mean_reward']:.2f} +/- {camram_results['std_reward']:.2f}")
    print(f"hybrid: {hybrid_results['mean_reward']:.2f} +/- {hybrid_results['std_reward']:.2f}")

    viper_to_hybrid = ((hybrid_results['mean_reward'] - viper_results['mean_reward']) /
                       abs(viper_results['mean_reward']) * 100)
    print(f"\nhybrid vs viper: {viper_to_hybrid:+.1f}%")


if __name__ == "__main__":
    main()
