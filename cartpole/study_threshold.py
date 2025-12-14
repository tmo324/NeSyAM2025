#!/usr/bin/env python3
"""
Parameter study - threshold and boost factor
"""
import sys
import os
import numpy as np
import pickle

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'PPO-PyTorch'))

from memory_augmented_ppo import load_ppo, eval_policy, collect_memory, SuccessMemoryCAM

env = "CartPole-v1"

print("loading ppo...")
ppo = load_ppo(env)

print("\ncollecting memory...")
base_mem = collect_memory(env, ppo, n_eps=50)
print(f"got {len(base_mem.states)} pairs")

print("\ntesting thresholds...")

# threshold sweep
print("\n--- Similarity Threshold ---")
thresholds = [0.3, 0.5, 0.7, 1.0]
thresh_results = {}

for t in thresholds:
    print(f"\nthreshold={t}...")
    mem = SuccessMemoryCAM(k_neighbors=5, similarity_threshold=t)
    mem.states = base_mem.states.copy()
    mem.actions = base_mem.actions.copy()

    r = eval_policy(env, ppo, n_eps=50, memory=mem, use_memory=True)
    thresh_results[t] = r
    print(f"  -> {r['success_rate']:.1f}%")

# boost sweep
print("\n\n--- Boost Factor ---")
boosts = [1.0, 1.3, 1.5, 2.0]
boost_results = {}

for b in boosts:
    print(f"\nboost={b}...")
    mem = SuccessMemoryCAM(k_neighbors=5, similarity_threshold=0.5)
    mem.states = base_mem.states.copy()
    mem.actions = base_mem.actions.copy()

    r = eval_policy(env, ppo, n_eps=50, memory=mem, use_memory=True, boost=b)
    boost_results[b] = r
    print(f"  -> {r['success_rate']:.1f}%")

print("\n--- results ---")

print("\nthreshold:")
for t in thresholds:
    r = thresh_results[t]
    print(f"  {t}: {r['success_rate']:.1f}%")

print("\nboost:")
for b in boosts:
    r = boost_results[b]
    print(f"  {b}x: {r['success_rate']:.1f}%")

best_t = max(thresholds, key=lambda x: thresh_results[x]['success_rate'])
best_b = max(boosts, key=lambda x: boost_results[x]['success_rate'])

print(f"\nbest: threshold={best_t}, boost={best_b}")

os.makedirs("results", exist_ok=True)
with open("results/threshold_study.pkl", 'wb') as f:
    pickle.dump({
        'thresholds': thresholds, 'thresh_results': thresh_results,
        'boosts': boosts, 'boost_results': boost_results
    }, f)
print("saved to results/threshold_study.pkl")
