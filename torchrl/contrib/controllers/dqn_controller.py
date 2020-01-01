import numpy as np
import torch
from torchrl.controllers import Controller
from torchrl.utils import ExpDecaySchedule
from torchrl.contrib.models import QNet


def epsilon_greedy(action_size: int, choices: np.array,
                   eps: float = 0.1):
  """
  Batched epsilon-greedy
  :param action_size: Total number of actions
  :param choices: A list of choices
  :param eps: Value of epsilon
  :return:
  """
  distribution = np.ones((len(choices), action_size),
                         dtype=np.float32) * eps / action_size
  distribution[np.arange(len(choices)), choices] += 1.0 - eps
  actions = np.array([
      np.random.choice(np.arange(action_size), p=dist)
      for dist in distribution
  ])
  return np.expand_dims(actions, axis=1)


class DQNController(Controller):
  def __init__(self, obs_size, action_size,
               double_dqn=False, gamma=.99, lr=1e-3,
               eps_max=1.0, eps_min=1e-2, n_eps_anneal=1000,
               n_update_interval=5, device=None):
    self.action_size = action_size
    self.device = device

    self.q_net = QNet(obs_size, action_size).to(self.device)
    self.target_q_net = QNet(obs_size, action_size).to(self.device)
    self.target_q_net.load_state_dict(self.q_net.state_dict())

    self.q_net_optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    self.double_dqn = double_dqn
    self.gamma = gamma
    self.n_update_interval = n_update_interval
    self.eps = ExpDecaySchedule(start=eps_max, end=eps_min,
                                num_steps=n_eps_anneal)

    self._steps = 0

  def act(self, obs):
    with torch.no_grad():
      obs_tensor = torch.from_numpy(
          np.array(obs)
      ).float().to(self.device).unsqueeze(0)

      actions = self.q_net(obs_tensor)

    actions = actions.argmax(dim=-1, keepdim=True).cpu().numpy()
    actions = epsilon_greedy(self.action_size, actions, self.eps.value)
    return actions.item()

  def learn(self, obs, action, reward, next_obs, done):
    self.q_net.train()

    current_q_values = self.q_net(obs).gather(1, action.long())

    with torch.no_grad():
      if self.double_dqn:
        _, next_actions = self.q_net(next_obs).max(1, keepdim=True)
        next_q_values = self.target_q_net(next_obs).gather(1, next_actions)
      else:
        next_q_values = self.target_q_net(next_obs)
        next_q_values = next_q_values.max(1)[0].unsqueeze(1)

      expected_q_values = reward + \
                          self.gamma * next_q_values * (1.0 - done.float())

    td_error = (expected_q_values - current_q_values)
    loss = td_error.pow(2).mean()

    self.q_net_optim.zero_grad()
    loss.backward()
    self.q_net_optim.step()

    self._steps += 1
    if self._steps % self.n_update_interval == 0:
      self.target_q_net.load_state_dict(self.q_net.state_dict())

    return dict(td_loss=loss.detach().cpu())
