import torch
from torchrl.controllers import Controller
from torchrl.contrib.models import QNet


class SACController(Controller):
  def __init__(self, obs_size, action_size, gamma=0.99,
               tau=0.1, lr=3e-4, device=None):
    self.device = device

    self.gamma = gamma
    self.tau = tau

    self.qnets = torch.nn.ModuleList([
        QNet(obs_size + action_size, 1, hidden_size=256),
        QNet(obs_size + action_size, 1, hidden_size=256)
    ]).to(self.device)
    self.qnets[1].load_state_dict(self.qnets[0].state_dict())

    self.qnet_optim = torch.optim.Adam(self.qnets.parameters(), lr=lr)

  def act(self, obs):
    raise NotImplementedError

  def learn(self):
    raise NotImplementedError
