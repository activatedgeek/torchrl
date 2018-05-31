import torch
import torch.nn as nn


class QNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNet, self).__init__()

        self._input_size = input_size
        self._output_size = output_size

        self.net = nn.Sequential(
            nn.Linear(self._input_size, 512),
            nn.ReLU(),
            nn.Linear(512, self._output_size)
        )

    def forward(self, obs):
        values = self.net(obs)
        return values

    def _init_weights(self):
        nn.init.xavier_uniform(self.net[0].weight)
        nn.init.xavier_uniform(self.net[2].weight)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # @TODO: add layer norm?
        self.net = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.action_size),
            nn.Tanh()
        )

        self._init_weights()

    def forward(self, obs):
        return self.net(obs)

    def _init_weights(self):
        nn.init.uniform(self.net[-2].weight, -3e-3, 3e-3)


class Critic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Critic, self).__init__()

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_size = hidden_size

        # @TODO: add layer norm?
        self.net = nn.Sequential(
            nn.Linear(self.state_size, self.hidden_size),
            nn.ReLU()
        )

        self.out = nn.Sequential(
            nn.Linear(self.hidden_size + self.action_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1)
        )

        self._init_weights()

    def forward(self, obs, action):
        x = self.net(obs)
        y = torch.cat([x, action], dim=1)
        out = self.out(y)
        return out

    def _init_weights(self):
        nn.init.uniform(self.out[-1].weight, -3e-4, 3e-4)


class ACNet(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(ACNet, self).__init__()

        self._input_size = input_size
        self._output_size = output_size
        self._hidden_size = hidden_size

        self.critic = nn.Sequential(
            nn.Linear(self._input_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(self._input_size, self._hidden_size),
            nn.ReLU(),
            nn.Linear(self._hidden_size, self._output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, obs):
        value = self.critic(obs)
        policy = self.actor(obs)
        return value, policy
