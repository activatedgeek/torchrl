import torch.nn as nn
from collections import OrderedDict


class SimplePolicyNet(nn.Module):
    """
    @WARN: This policy network is only for demonstration of I/O
    """
    def __init__(self, input_size, output_size):
        super(SimplePolicyNet, self).__init__()

        self._input_size = input_size
        self._output_size = output_size

        self.net = nn.Sequential(OrderedDict([
            ('f1', nn.Linear(self._input_size, 512)),
            ('relu1', nn.ReLU()),
            ('f2', nn.Linear(512, self._output_size)),
        ]))

        self.value = nn.Linear(self._output_size, 1)
        self.policy = nn.Softmax()

    def forward(self, obs):
        values = self.net(obs)
        value = self.value(values)
        policy = self.policy(values)
        return value, policy.data
