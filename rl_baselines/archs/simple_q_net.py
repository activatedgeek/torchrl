import torch.nn as nn
from collections import OrderedDict


class SimpleQNet(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleQNet, self).__init__()

        self._input_size = input_size
        self._output_size = output_size

        self.net = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(self._input_size, 512)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(512, self._output_size)),
        ]))
        nn.init.xavier_uniform(self.net[0].weight)
        nn.init.xavier_uniform(self.net[2].weight)

    def forward(self, obs):
        values = self.net(obs)
        return values
