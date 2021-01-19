from utils.maze_env import Maze
from utils.rl_brain import DQN


class Run:
    def __init__(self, RL, env, episodes):
        self.RL = RL
        self.env = env
        self.episodes = episodes

    def __call__(self):
        step = 0

        for episode in range(self.episodes):
            observation = self.env.reset()

            while True:
                self.env.render()

                action = self.RL.choose_action(observation)

                observation_, reward, done = self.env.step(action)

                self.RL.store_transition(observation, action, reward,
                                         observation_)

                if step > 200 and step % 5 == 0:
                    self.RL.learn()

                observation = observation_

                if done:
                    break

                step += 1

        print('Game Over')
        self.env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = DQN(env.n_actions, env.n_features, learning_rate=0.01,
             reward_decay=0.9, e_greedy=0.9, replace_target_iter=200,
             memory_size=2000)
    update = Run(RL, env, 100)
    env.after(100, update())
    env.mainloop()
    RL.plot_cost()
