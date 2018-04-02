import torch
from torch.autograd import Variable
from . import BaseAgent


class DQN(BaseAgent):
    def __init__(self, qnet):
        super(DQN, self).__init__()

        self.qnet = qnet

    def forward(self, var):
        return self.qnet(var)

    def act(self, state):
        action_values = self.forward(Variable(torch.FloatTensor([state]), volatile=True))
        value, action = action_values.max(1)
        return action.data[0]
