# Need a replay buffer class
# Need a class for a target Q network (function of s, a)
# Use batch norm
# The policy is deterministic, how to handle explore and exploit?
# We have two actor and two critic networks, a target for each.
# Update are soft, according to theta_prime=tau*theta+(1-tau)*theta_prime

import torch
import torch.autograd
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

from models import Actor, Critic
from utils import OUNoise, Memory


class DDPGAgent:
    def __init__(self, env, hidden_size=256, actor_lr=1e-4, critic_lr=1e-3,
                 gamma=.99, tau=1e-2, max_memory_size=50000):
        self.num_states = env.observation_space.shape[0]
        self.num_actions = env.action_space.shape[0]
        self.gamma = gamma
        self.tau = tau

        # Init networks
        self.actor = Actor(self.num_states, hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, hidden_size,
                                  self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, hidden_size,
                             self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions,
                                    hidden_size, self.num_actions)

        # Copy parameters to target networks
        for target_param, param in zip(self.actor_target.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.critic_target.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(param.data)

        # Training
        self.memory = Memory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),
                                           lr=critic_lr)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0, 0]
        return action

    def update(self, batch_size):
        states, actions, rewards, next_states, _ = self.memory.sample(
            batch_size)
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        # Critic loss
        Qvals = self.critic.forward(states, actions)
        next_actions = self.actor_target.forward(next_states)
        next_Q = self.critic_target.forward(next_states, next_actions.detach())
        Qprime = rewards + self.gamma * next_Q
        critic_loss = self.critic_criterion(Qvals, Qprime)

        # Actor loss
        policy_loss = -self.critic.forward(states,
                                           self.actor.forward(states)).mean()

        # Update networks
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update target networks
        for target_param, param in zip(self.actor_target.parameters(),
                                       self.actor.parameters()):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1 - self.tau))
        for target_param, param in zip(self.critic_target.parameters(),
                                       self.critic.parameters()):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1 - self.tau))
