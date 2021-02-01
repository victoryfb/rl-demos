import gym
import matplotlib.pyplot as plt
import numpy as np

from Agent import VanillaAgent


agent = VanillaAgent(lr=0.0005, gamma=0.99, n_actions=4, layer1_size=256,
                     layer2_size=256)

env = gym.make('LunarLander-v2')
score_history = []

num_episodes = 2000

for i in range(num_episodes):
    done = False
    score = 0
    observation = env.reset()
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, done, info = env.step(action)
        agent.store_transition(observation, action, reward)
        observation = observation_
        score += reward
    score_history.append(score)
    agent.learn()

    print('episode: ', i, 'score: %.1f' % score,
          'average score %.1f' % np.mean(
              score_history[max(0, i - 100):(i + 1)]))

filename = 'lunar-lander-keras-64x64-alpha0005-2000games.png'
window = 100
N = len(score_history)
running_avg = np.empty(N)
for t in range(N):
    running_avg[t] = np.mean(score_history[max(0, t - window):(t + 1)])
x = [i for i in range(N)]
plt.ylabel('Score')
plt.xlabel('Game')
plt.plot(x, running_avg)
plt.savefig(filename)
