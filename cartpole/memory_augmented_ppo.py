#!/usr/bin/env python3
"""
Memory-Augmented PPO - k-NN memory lookup to boost action selection
"""
import sys
import os
import numpy as np
import torch
import gymnasium as gym
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PPO-PyTorch'))
from PPO import PPO


class SuccessMemoryCAM:
    def __init__(self, k_neighbors=5, similarity_threshold=0.5):
        self.k = k_neighbors
        self.threshold = similarity_threshold
        self.states = []
        self.actions = []

    def store(self, state, action):
        self.states.append(state.copy())
        self.actions.append(action)

    def lookup(self, query_state):
        if len(self.states) == 0:
            return -1, float('inf')

        states_arr = np.array(self.states)
        dists = np.linalg.norm(states_arr - query_state, axis=1)

        k = min(self.k, len(dists))
        k_idx = np.argpartition(dists, k-1)[:k]
        k_dists = dists[k_idx]
        k_actions = [self.actions[i] for i in k_idx]

        best_action = max(set(k_actions), key=k_actions.count)
        min_dist = k_dists.min()

        return best_action, min_dist

    def size(self):
        return len(self.states)


class MemoryAugmentedPPO:
    def __init__(self, ppo_agent, success_memory, boost_factor=1.3):
        self.ppo = ppo_agent
        self.memory = success_memory
        self.boost = boost_factor
        self.hits = 0
        self.misses = 0

    def select_action(self, state, use_memory=True):
        if not use_memory:
            return self.ppo.select_action(state)

        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(self.ppo.policy.actor[0].weight.device)
            probs = self.ppo.policy.actor(state_t).cpu().numpy()

        mem_action, dist = self.memory.lookup(state)

        if mem_action >= 0 and dist < self.memory.threshold:
            probs[mem_action] *= self.boost
            probs /= probs.sum()
            self.hits += 1
        else:
            self.misses += 1

        return np.random.choice(len(probs), p=probs)

    def get_stats(self):
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            'memory_size': self.memory.size(),
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate
        }


def load_ppo(env_name, model_path=None):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    ppo = PPO(state_dim, action_dim,
              lr_actor=0.0003, lr_critic=0.001,
              gamma=0.99, K_epochs=40, eps_clip=0.2,
              has_continuous_action_space=False,
              action_std_init=None)

    if model_path is None:
        model_path = f"PPO-PyTorch/ppo_{env_name.split('-')[0].lower()}.pth"

    print(f"loading {model_path}")
    ppo.load(model_path)
    return ppo


def eval_policy(env_name, ppo_agent, n_eps=100, memory=None, use_memory=False, boost=1.3):
    env = gym.make(env_name)

    if memory and use_memory:
        agent = MemoryAugmentedPPO(ppo_agent, memory, boost_factor=boost)
        mode = "augmented"
    else:
        agent = ppo_agent
        mode = "baseline"

    print(f"\nrunning {mode} on {env_name}...")

    rewards = []
    lengths = []
    successes = 0

    for ep in range(n_eps):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        ep_reward = 0
        ep_len = 0
        done = False

        while not done:
            if isinstance(agent, MemoryAugmentedPPO):
                action = agent.select_action(state, use_memory=use_memory)
            else:
                action = agent.select_action(state)

            result = env.step(action)
            if len(result) == 5:
                state, reward, term, trunc, _ = result
                done = term or trunc
            else:
                state, reward, done, _ = result

            ep_reward += reward
            ep_len += 1

        rewards.append(ep_reward)
        lengths.append(ep_len)

        if env_name == "CartPole-v1":
            success = ep_reward >= 450
        elif "LunarLander" in env_name:
            success = ep_reward >= 200
        else:
            success = ep_reward > 0

        if success:
            successes += 1

        if (ep + 1) % 20 == 0:
            print(f"  ep {ep+1}: avg={np.mean(rewards[-20:]):.1f}, success={successes}/{ep+1}")

    env.close()

    results = {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'success_rate': 100.0 * successes / n_eps,
        'mean_length': np.mean(lengths),
    }

    print(f"  reward: {results['mean_reward']:.1f} +/- {results['std_reward']:.1f}")
    print(f"  success: {results['success_rate']:.1f}%")

    if isinstance(agent, MemoryAugmentedPPO):
        stats = agent.get_stats()
        print(f"  mem size: {stats['memory_size']}, hit rate: {100*stats['hit_rate']:.1f}%")
        results['memory_stats'] = stats

    return results


def collect_memory(env_name, ppo_agent, n_eps=50):
    print(f"\ncollecting memory from {env_name}...")

    env = gym.make(env_name)
    memory = SuccessMemoryCAM(k_neighbors=5, similarity_threshold=0.5)

    if env_name == "CartPole-v1":
        thresh = 450
    elif "LunarLander" in env_name:
        thresh = 200
    else:
        thresh = 0

    success_count = 0
    total_eps = 0

    while success_count < n_eps and total_eps < n_eps * 5:
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        ep_states = []
        ep_actions = []
        ep_reward = 0
        done = False

        while not done:
            action = ppo_agent.select_action(state)
            ep_states.append(state.copy())
            ep_actions.append(action)

            result = env.step(action)
            if len(result) == 5:
                state, reward, term, trunc, _ = result
                done = term or trunc
            else:
                state, reward, done, _ = result
            ep_reward += reward

        total_eps += 1

        if ep_reward >= thresh:
            for s, a in zip(ep_states, ep_actions):
                memory.store(s, a)
            success_count += 1

            if success_count % 10 == 0:
                print(f"  {success_count}/{n_eps} episodes, {memory.size()} pairs")

    env.close()

    print(f"  done: {memory.size()} pairs")
    return memory


def run_experiment(env_name, n_eval=100, n_mem=50):
    print(f"\n--- {env_name} ---")

    ppo = load_ppo(env_name)
    baseline = eval_policy(env_name, ppo, n_eval, memory=None, use_memory=False)
    memory = collect_memory(env_name, ppo, n_mem)
    augmented = eval_policy(env_name, ppo, n_eval, memory=memory, use_memory=True)

    b_sr = baseline['success_rate']
    a_sr = augmented['success_rate']
    print(f"\nbaseline: {b_sr:.1f}% -> augmented: {a_sr:.1f}% (+{a_sr-b_sr:.1f}%)")

    return {
        'env': env_name,
        'baseline': baseline,
        'augmented': augmented,
        'memory_size': memory.size()
    }


def main():
    results = {}

    results['CartPole-v1'] = run_experiment("CartPole-v1", n_eval=100, n_mem=50)
    results['LunarLander-v3'] = run_experiment("LunarLander-v3", n_eval=100, n_mem=50)

    os.makedirs("results", exist_ok=True)
    with open("results/experiment_results.pkl", 'wb') as f:
        pickle.dump(results, f)

    print("\n--- summary ---")
    for env, r in results.items():
        b = r['baseline']['success_rate']
        a = r['augmented']['success_rate']
        print(f"{env}: {b:.1f}% -> {a:.1f}%")


if __name__ == "__main__":
    main()
