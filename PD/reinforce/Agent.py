import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp


class PolicyGradientNetwork(Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1 = Dense(fc1_dims, activation='relu')
        self.fc2 = Dense(fc2_dims, activation='relu')
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state, **kwargs):
        value = self.fc1(state)
        value = self.fc2(value)
        probs = self.pi(value)
        return probs


class Agent:
    def __init__(self, lr, gamma, n_actions, layer1_size=256, layer2_size=256,
                 filename='reinforce.h5'):
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.model_file = filename

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

        self.policy = PolicyGradientNetwork(n_actions, layer1_size,
                                            layer2_size)
        self.policy.compile(optimizer=Adam(learning_rate=lr))

    def store_transition(self, observation, action, reward):
        """
        Save transition including current state, action, reward.

        Parameters
        ----------
        observation: ndarray
            Current state
        action: float
            Predicted action based on current state
        reward: float
            Reward of the predicted action
        """
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def choose_action(self, observation):
        """
        Choose action from action space based on the probability of each action
        which is the prediction of the model.

        Parameters
        ----------
        observation: ndarray
            Current state

        Returns
        -------
        action: int
            Chosen action

        """
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        return action.numpy()[0]

    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = np.array(self.reward_memory)

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(
            zip(gradient, self.policy.trainable_variables))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

    def save(self):
        self.policy.save(self.model_file)

    def load(self):
        self.policy = load_model(self.model_file)
