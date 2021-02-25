<<<<<<< HEAD:Policy-Gradient/reinforce/Agent.py
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
=======
import os

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


class VanillaAgent:
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


class ActorCriticNetwork(Model):
    def __init__(self, n_actions, fc1_dims=1024, fc2_dims=512,
                 name='actor_critic', ckpts_dir='./checkpoints'):
        super(ActorCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = ckpts_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        self.fc1 = Dense(self.fc1_dims, activation='relu')
        self.fc2 = Dense(self.fc2_dims, activation='relu')
        self.v = Dense(1, activation=None)
        self.pi = Dense(n_actions, activation='softmax')

    def call(self, state, **kwargs):
        value = self.fc1(state)
        value = self.fc2(value)

        v = self.v(value)
        pi = self.pi(value)

        return v, pi


class ActorCriticAgent:
    def __init__(self, alpha=0.0003, gamma=0.99, n_actions=2, fc1_dims=1024,
                 fc2_dims=512, name='actor_critic', ckpts_dir='./checkpoints'):
        self.gamma = gamma
        self.n_actions = n_actions
        self.action_space = [i for i in range(n_actions)]

        self.actor_critic = ActorCriticNetwork(n_actions, fc1_dims, fc2_dims,
                                               name, ckpts_dir)

        self.actor_critic.compile(optimizer=Adam(learning_rate=alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        _, probs = self.actor_critic(state)

        action_probabilities = tfp.distributions.Categorical(probs=probs)
        action = action_probabilities.sample()

        return action.numpy()[0]

    def save_models(self):
        self.actor_critic.save_weights(self.actor_critic.checkpoint_file)

    def load_models(self):
        self.actor_critic.load_weights(self.actor_critic.checkpoint_file)

    def learn(self, state, action, reward, state_, done):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        state_ = tf.convert_to_tensor([state_], dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        with tf.GradientTape(persistent=True) as tape:
            state_value, probs = self.actor_critic(state)
            state_value_, _ = self.actor_critic(state_)
            state_value = tf.squeeze(state_value)
            state_value_ = tf.squeeze(state_value_)

            action_probs = tfp.distributions.Categorical(probs=probs)
            log_prob = action_probs.log_prob(action)

            delta = reward + self.gamma * state_value_ * (
                        1 - int(done)) - state_value
            actor_loss = -log_prob * delta
            critic_loss = delta ** 2
            total_loss = actor_loss + critic_loss

        gradient = tape.gradient(total_loss,
                                 self.actor_critic.trainable_variables)
        self.actor_critic.optimizer.apply_gradients(zip(
            gradient, self.actor_critic.trainable_variables))
>>>>>>> f6d8081a9291eed762277a0c9deb7ab208d08d65:Policy-Gradient/Agent.py
