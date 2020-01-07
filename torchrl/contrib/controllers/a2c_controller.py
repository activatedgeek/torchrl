import torch
import numpy as np
from torchrl.controllers import Controller
from torchrl.contrib.models import A2CNet


class A2CController(Controller):
  def __init__(self, obs_size, action_size,
               gamma=0.99, lmbda=1.0, lr=1e-3,
               alpha=0.5, beta=1.0, device=None):
    self.device = device

    self.ac_net = A2CNet(obs_size, action_size, 256).to(self.device)
    self.ac_net_optim = torch.optim.Adam(self.ac_net.parameters(), lr=lr)

    self.gamma = gamma
    self.lmbda = lmbda
    self.alpha = alpha
    self.beta = beta

  def act(self, obs):
    with torch.no_grad():
      obs_tensor = torch.Tensor(obs).float().to(self.device)
      _, dist = self.ac_net(obs_tensor)

    action = dist.sample().cpu().numpy()
    return action

  def compute_return(self, obs, action, reward, next_obs, done):  # pylint: disable=unused-argument
    with torch.no_grad():
      values, _ = self.ac_net(obs)
      if not done[-1]:
        next_value, _ = self.ac_net(next_obs[-1:])
        values = torch.cat([values, next_value], dim=0)
      else:
        values = torch.cat([values, values.new_tensor(np.zeros((1, 1)))], dim=0)

      returns = reward.new_tensor(np.zeros((len(reward), 1)))
      gae = 0.0
      for step in reversed(range(len(reward))):
        delta = reward[step] + self.gamma * values[step + 1] - values[step]
        gae = delta + self.gamma * self.lmbda * gae
        returns[step] = gae + values[step]

      return returns

  def learn(self, obs, action, reward, next_obs, done, returns):  # pylint: disable=unused-argument
    values, dist = self.ac_net(obs)

    advantages = returns - values

    action_log_probs = dist.log_prob(action.squeeze(-1)).unsqueeze(1)
    actor_loss = - (advantages.detach() * action_log_probs).mean()

    critic_loss = torch.nn.functional.mse_loss(values, returns)

    entropy_loss = dist.entropy().mean()

    loss = actor_loss + self.alpha * critic_loss - self.beta * entropy_loss

    self.ac_net_optim.zero_grad()
    loss.backward()
    self.ac_net_optim.step()

    return dict(
        actor_loss=actor_loss.detach(),
        critic_loss=critic_loss.detach(),
        entropy_loss=entropy_loss.detach()
    )
