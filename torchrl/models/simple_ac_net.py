import torch.nn as nn
from collections import OrderedDict


class SimpleACNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleACNet, self).__init__()

        self._input_size = input_size
        self._output_size = output_size

        self.net = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(self._input_size, 64)),
            ('relu1', nn.ReLU()),
        ]))

        self.actor = nn.Sequential(OrderedDict([
            ('f2', nn.Linear(64, output_size)),
            ('smax1', nn.Softmax(dim=1)),
        ]))
        self.critic = nn.Linear(64, 1)

    def forward(self, obs):
        h = self.net(obs)
        value = self.critic(h)
        policy = self.actor(h)
        return value, policy
