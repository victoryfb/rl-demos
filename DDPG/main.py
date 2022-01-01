import sys

import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ddpg import DDPGAgent
from utils import NormalizedEnv, OUNoise

# env = NormalizedEnv(gym.make("Pendulum-v1"))
env = gym.make("Pendulum-v1")
agent = DDPGAgent(env)
noise = OUNoise(env.action_space)
batch_size = 128
rewards = []
avg_rewards = []

for episode in range(50):
    state = env.reset()
    episode_reward = 0

    for step in range(500):
        action = agent.get_action(state)
        action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action)
        agent.memory.push(state, action, reward, new_state, done)

        if len(agent.memory) > batch_size:
            agent.update(batch_size)

        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write(
                "episode: {}, reward: {}, average_reward: {} \n".format(
                    episode, np.round(episode_reward, decimals=2),
                    np.mean(rewards[-10:])))
            break
    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.figure(figsize=(8, 6))
plt.plot(avg_rewards, label="Average Rewards")
plt.plot(rewards, label="Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.tight_layout()
plt.savefig("result.pdf")