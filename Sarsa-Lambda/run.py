from utils.maze_env import Maze
from utils.rl_brain import SarsaLambdaTable


class Run:
    def __init__(self, RL, env, episodes):
        self.RL = RL
        self.env = env
        self.episodes = episodes

    def __call__(self):
        for episode in range(self.episodes):
            # init observation
            observation = self.env.reset()

            # choose action
            action = self.RL.choose_action(str(observation))

            while True:
                self.env.render()

                # take the action and get next observation and reward
                observation_, reward, done = self.env.step(action)

                # choose action
                action_ = self.RL.choose_action(str(observation_))

                # learn
                self.RL.learn(str(observation), action, reward,
                              str(observation_), action_)

                # update observation and action
                observation = observation_
                action = action_

                if done:
                    break

        print('game over')
        self.env.destroy()


if __name__ == '__main__':
    env = Maze()
    RL = SarsaLambdaTable(actions=list(range(env.n_actions)))

    update = Run(RL, env, 100)
    env.after(100, update())
    env.mainloop()
