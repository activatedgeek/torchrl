from torchrl.controllers import Controller
from torchrl.contrib.models import QNet


class SACController(Controller):
  def __init__(self, obs_size, action_size, gamma=0.99, tau=0.1,
               device=None):
    self.device = device

    self.gamma = gamma
    self.tau = tau

    self.qnet1 = QNet(obs_size + action_size, 1,
                      hidden_size=256).to(self.device)
    self.qnet2 = QNet(obs_size + action_size, 1,
                      hidden_size=256).to(self.device)
    self.qnet2.load_state_dict(self.qnet1.state_dict())

  def act(self, obs):
    raise NotImplementedError

  def learn(self):
    raise NotImplementedError
