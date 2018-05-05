import torch
import torch.nn as nn


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
