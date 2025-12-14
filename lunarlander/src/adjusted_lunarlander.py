"""
LunarLander wrapper with adjusted crash penalty
"""
import gymnasium as gym
from gymnasium import Wrapper
import numpy as np


class AdjustedLunarLander(Wrapper):
    def __init__(self, env_name="LunarLander-v2", crash_penalty=-30, **kwargs):
        base_env = gym.make(env_name, **kwargs)
        super().__init__(base_env)

        self.original_crash_penalty = -100
        self.new_crash_penalty = crash_penalty
        self.last_shaping = None

    def reset(self, **kwargs):
        self.last_shaping = None
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)

        # If episode terminated (not truncated), check if it was a crash or landing
        if terminated and not truncated:
            # LunarLander gives +100 for safe landing (both legs down)
            # and -100 for crash (body touches ground)
            #
            # Since we can't directly detect crash vs landing from the return values,
            # we use a heuristic: if reward is highly negative (< -50), it was a crash
            if reward < -50:
                # This was a crash with -100 penalty, adjust it
                reward = reward - self.original_crash_penalty + self.new_crash_penalty
                info['adjusted_crash_penalty'] = True
            elif reward > 50:
                # This was a successful landing (+100), keep it
                info['successful_landing'] = True

        return state, reward, terminated, truncated, info


def make_adjusted_lunarlander(crash_penalty=-30):
    return AdjustedLunarLander("LunarLander-v2", crash_penalty=crash_penalty)


if __name__ == "__main__":
    env = make_adjusted_lunarlander(crash_penalty=-30)
    print("testing...")

    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        while not done:
            action = env.action_space.sample()
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            steps += 1

            if steps > 1000:
                break

        status = "crashed" if episode_reward < 0 else "landed"
        print(f"  ep {episode + 1}: {episode_reward:.2f} ({steps} steps, {status})")

    print("done")
