import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


class ReplayBuffer:
    def __init__(self, max_size, input_dims, n_actions, discrete=True):
        self.mem_size = max_size
        self.mem_counter = 0
        self.discrete = discrete

        self.state_memory = np.zeros((self.mem_size, *input_dims),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims),
                                         dtype=np.float32)
        a_type = np.int8 if self.discrete else np.float32
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=a_type)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.int32)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        if self.discrete:
            actions = np.zeros(self.action_memory.shape[1])
            actions[action] = 1.0
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
        self.terminal_memory[index] = 1 - int(done)
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


def build_dqn(lr, n_actions, fc1_dims, fc2_dims):
    model = Sequential([
        Dense(fc1_dims, activation='relu'),
        Dense(fc2_dims, activation='relu'),
        Dense(n_actions, activation=None)
    ])
    model.compile(optimizer=Adam(learning_rate=lr), loss='mse')

    return model


class Agent:
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims,
                 epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000,
                 filename='model.ht'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = filename
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.q_eval = build_dqn(lr, n_actions, 256, 256)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)
            action = np.argmax(actions)

        return action

    def learn(self):
        pass

    def save_model(self):
        self.q_eval.save(self.model_file)

    def load_model(self):
        self.q_eval = load_model(self.model_file)


class DQNAgent(Agent):
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims,
                 epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000,
                 filename='dqn_model.ht'):
        super(DQNAgent, self).__init__(lr, gamma, n_actions, epsilon,
                                       batch_size, input_dims,
                                       epsilon_dec, epsilon_end,
                                       mem_size, filename)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size)
        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)

        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size)

        q_target[batch_index, actions] = rewards \
            + self.gamma * np.max(q_next, axis=1) * dones
        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min


class DDQNAgent(Agent):
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims,
                 epsilon_dec=1e-3, epsilon_end=0.01, mem_size=1000000,
                 filename='dqn_model.ht', replace_target=100):
        super(DDQNAgent, self).__init__(lr, gamma, n_actions, epsilon,
                                        batch_size, input_dims,
                                        epsilon_dec, epsilon_end,
                                        mem_size, filename)
        self.replace_target = replace_target
        self.q_target = build_dqn(lr, n_actions, 256, 256)

    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size)

        action_values = np.array(self.action_space, dtype=np.int8)
        action_indices = np.dot(actions, action_values)

        q_next = self.q_target.predict(states_)
        q_eval = self.q_eval.predict(states_)
        q_pred = self.q_eval.predict(states)

        max_actions = np.argmax(q_eval, axis=1)
        q_target = np.copy(q_pred)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, action_indices] = rewards \
            + self.gamma * q_next[batch_index, max_actions.astype(int)] * dones

        self.q_eval.train_on_batch(states, q_target)
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

        if self.memory.mem_counter % self.replace_target == 0:
            self._update_network_parameters()

    def _update_network_parameters(self):
        self.q_target.set_weights(self.q_eval.get_weights())

    def load_model(self):
        self.q_eval = load_model(self.model_file)
        # if we are in evaluation mode we want to use the best weights for
        # q_target
        if self.epsilon == 0.0:
            self._update_network_parameters()
