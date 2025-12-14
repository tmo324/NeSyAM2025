#!/usr/bin/env python3
"""
Train PPO, VIPER, CAM-RAM, and Hybrid on LunarLander-v3
"""
import os
import sys
import yaml
import pickle
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from symbolic.viper_simple import VIPERAgent
from memory.camram_agent import CAMRAM


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), 'configs', 'lunarlander.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_ppo(config):
    print("\n--- 1/4: ppo ---")

    env_name = config['environment']['name']
    seed = config['environment']['seed']

    env = gym.make(env_name)
    env.reset(seed=seed)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=config['ppo']['learning_rate'],
        n_steps=config['ppo']['n_steps'],
        batch_size=config['ppo']['batch_size'],
        n_epochs=config['ppo']['n_epochs'],
        gamma=config['ppo']['gamma'],
        gae_lambda=config['ppo']['gae_lambda'],
        clip_range=config['ppo']['clip_range'],
        ent_coef=config['ppo']['ent_coef'],
        vf_coef=config['ppo']['vf_coef'],
        max_grad_norm=config['ppo']['max_grad_norm'],
        verbose=1,
        seed=seed
    )

    print(f"training for {config['training']['total_timesteps']} timesteps...")
    model.learn(total_timesteps=config['training']['total_timesteps'])

    output_dir = os.path.join(os.path.dirname(__file__), 'weights')
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, 'ppo_model.zip')
    model.save(model_path)
    print(f"saved to {model_path}")

    print("evaluating...")
    eval_env = gym.make(env_name)
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=config['training']['n_eval_episodes'], deterministic=True
    )
    eval_env.close()

    results = {'mean_reward': float(mean_reward), 'std_reward': float(std_reward), 'env_name': env_name}

    with open(os.path.join(output_dir, 'ppo_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"ppo: {mean_reward:.2f} +/- {std_reward:.2f}")

    env.close()
    return model, results


def extract_viper(ppo_model, config):
    print("\n--- 2/4: viper ---")

    env_name = config['environment']['name']
    seed = config['environment']['seed']

    env = gym.make(env_name)
    env.reset(seed=seed)

    print("collecting expert trajectories...")
    states = []
    actions = []

    for episode in range(config['viper']['n_batch_rollouts']):
        state, _ = env.reset()
        done = False

        while not done:
            action, _ = ppo_model.predict(state, deterministic=True)
            states.append(state)
            actions.append(action)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        if (episode + 1) % 10 == 0:
            print(f"  {episode + 1}/{config['viper']['n_batch_rollouts']} episodes")

    states = np.array(states)
    actions = np.array(actions)

    print(f"got {len(states)} pairs")
    viper_agent = VIPERAgent(
        max_depth=config['viper']['max_depth'],
        min_samples_split=config['viper']['min_samples_split']
    )
    viper_agent.train(states, actions)

    print("evaluating...")
    eval_rewards = []
    for _ in range(config['viper']['n_test_rollouts']):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = viper_agent.predict(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        eval_rewards.append(episode_reward)

    results = {
        'mean_reward': float(np.mean(eval_rewards)),
        'std_reward': float(np.std(eval_rewards)),
        'env_name': env_name,
        'tree_depth': viper_agent.get_depth(),
        'num_leaves': viper_agent.get_num_leaves()
    }

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'weights')
    agent_path = os.path.join(output_dir, 'viper_agent.pkl')
    viper_agent.save(agent_path)

    with open(os.path.join(output_dir, 'viper_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"viper: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"  depth={results['tree_depth']}, leaves={results['num_leaves']}")

    env.close()
    return viper_agent, results


def train_camram(config):
    print("\n--- 3/4: camram ---")

    env_name = config['environment']['name']
    seed = config['environment']['seed']

    env = gym.make(env_name)
    env.reset(seed=seed)

    agent = CAMRAM(
        num_actions=env.action_space.n,
        k=config['camram']['k_neighbors'],
        max_mem=config['camram']['max_memory_per_action'],
        epsilon=config['camram']['epsilon'],
        eps_decay=config['camram']['epsilon_decay'],
        eps_min=config['camram']['epsilon_min'],
        gamma=config['camram']['gamma'],
        lr=config['camram']['learning_rate']
    )

    total_episodes = config['camram']['total_episodes']
    print(f"training for {total_episodes} episodes...")

    for episode in range(total_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = agent.predict(state, deterministic=False)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.update(state, action, reward, next_state, done)

            state = next_state
            episode_reward += reward

        if (episode + 1) % 100 == 0:
            print(f"  ep {episode + 1}/{total_episodes}")

    print("evaluating...")
    eval_rewards = []
    for _ in range(config['camram']['n_eval_episodes']):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _ = agent.predict(state, deterministic=True)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        eval_rewards.append(episode_reward)

    results = {
        'mean_reward': float(np.mean(eval_rewards)),
        'std_reward': float(np.std(eval_rewards)),
        'env_name': env_name
    }

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'weights')
    agent_path = os.path.join(output_dir, 'camram_agent.pkl')
    agent.save(agent_path)

    with open(os.path.join(output_dir, 'camram_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"camram: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")

    env.close()
    return agent, results


def evaluate_hybrid(viper_agent, camram_agent, config):
    print("\n--- 4/4: hybrid ---")

    env_name = config['environment']['name']
    env = gym.make(env_name)

    eval_rewards = []
    viper_selections = 0
    memory_selections = 0
    confidence_threshold = config['hybrid']['confidence_threshold']

    for episode in range(config['training']['n_eval_episodes']):
        state, _ = env.reset()
        episode_reward = 0
        done = False

        while not done:
            viper_action = viper_agent.predict(state)
            memory_action, confidence = camram_agent.predict(state, deterministic=True)

            if confidence > confidence_threshold:
                action = memory_action
                memory_selections += 1
            else:
                action = viper_action
                viper_selections += 1

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

        eval_rewards.append(episode_reward)

    total_decisions = viper_selections + memory_selections
    results = {
        'mean_reward': float(np.mean(eval_rewards)),
        'std_reward': float(np.std(eval_rewards)),
        'env_name': env_name,
        'viper_percentage': (viper_selections / total_decisions * 100) if total_decisions > 0 else 0,
        'camram_percentage': (memory_selections / total_decisions * 100) if total_decisions > 0 else 0
    }

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), 'weights')
    with open(os.path.join(output_dir, 'hybrid_results.pkl'), 'wb') as f:
        pickle.dump(results, f)

    print(f"hybrid: {results['mean_reward']:.2f} +/- {results['std_reward']:.2f}")
    print(f"  viper: {results['viper_percentage']:.1f}%, memory: {results['camram_percentage']:.1f}%")

    env.close()
    return results


def main():
    print("training all approaches on lunarlander-v3\n")

    # Load config
    config = load_config()

    # Train all approaches
    ppo_model, ppo_results = train_ppo(config)
    viper_agent, viper_results = extract_viper(ppo_model, config)
    camram_agent, camram_results = train_camram(config)
    hybrid_results = evaluate_hybrid(viper_agent, camram_agent, config)

    print("\n--- done ---")
    print(f"ppo:    {ppo_results['mean_reward']:.2f} +/- {ppo_results['std_reward']:.2f}")
    print(f"viper:  {viper_results['mean_reward']:.2f} +/- {viper_results['std_reward']:.2f}")
    print(f"camram: {camram_results['mean_reward']:.2f} +/- {camram_results['std_reward']:.2f}")
    print(f"hybrid: {hybrid_results['mean_reward']:.2f} +/- {hybrid_results['std_reward']:.2f}")
    print("\nmodels saved to weights/")


if __name__ == "__main__":
    main()
