import torch
import gymnasium as gym
from PPO import PPO
import matplotlib.pyplot as plt

# ---------------- CONFIG ---------------- #
env_name = "CartPole-v1"
has_continuous_action_space = False  # CartPole has a discrete action space

max_ep_len = 500                     # Max timesteps per episode
max_training_timesteps = int(1e5)    # Total timesteps for training (updated to 50k)

print_freq = max_ep_len * 4          # Print every N timesteps
save_model_path = "ppo_cartpole.pth" # Model checkpoint path
reward_plot_path = "cartpole_rewards.png"

K_epochs = 80                        # PPO epochs
eps_clip = 0.2                       # PPO clip parameter
gamma = 0.99                         # Discount factor

lr_actor = 0.0003                    # Learning rate for actor
lr_critic = 0.001                    # Learning rate for critic

# Gymnasium environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

# Initialize PPO Agent
ppo_agent = PPO(
    state_dim=state_dim,
    action_dim=action_dim,
    lr_actor=lr_actor,
    lr_critic=lr_critic,
    gamma=gamma,
    K_epochs=K_epochs,
    eps_clip=eps_clip,
    has_continuous_action_space=has_continuous_action_space
)

# Start training
print(f"training on {env_name}")
time_step = 0
i_episode = 0
episode_rewards = []

while time_step <= max_training_timesteps:
    state, _ = env.reset()
    current_ep_reward = 0

    for t in range(1, max_ep_len + 1):
        action = ppo_agent.select_action(state)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Store reward and terminal flag
        ppo_agent.buffer.rewards.append(reward)
        ppo_agent.buffer.is_terminals.append(done)

        time_step += 1
        current_ep_reward += reward

        if done:
            break

    # Update PPO policy
    ppo_agent.update()

    episode_rewards.append(current_ep_reward)

    if i_episode % 10 == 0:
        print(f"Episode: {i_episode}\tTimestep: {time_step}\tReward: {current_ep_reward:.2f}")

    i_episode += 1

env.close()
print("done")

# Save the model
ppo_agent.save(save_model_path)
print(f"saved to {save_model_path}")

# Plot and save rewards
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("PPO on CartPole-v1")
plt.grid(True)
plt.savefig(reward_plot_path)
print(f"plot saved to {reward_plot_path}")
