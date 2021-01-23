from maze_env import Maze
from rl_brain import QLearningTable


class Run:
    def __init__(self, RL, env, episodes):
        self.RL = RL
        self.env = env
        self.episodes = episodes

    def __call__(self):
        for episode in range(self.episodes):
            # init observation
            observation = self.env.reset()

            while True:
                self.env.render()

                # choose action
                action = self.RL.choose_action(str(observation))

                # take the action and get next observation and reward
                observation_, reward, done = self.env.step(action)

                # learn
                self.RL.learn(str(observation), action, reward,
                              str(observation_))

                # update observation
                observation = observation_

                if done:
                    break

        print('game over')
        self.env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))

    update = Run(RL, env, 100)
    env.after(100, update())
    env.mainloop()
