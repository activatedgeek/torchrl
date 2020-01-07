import numpy as np
import torch
from torchrl.controllers import Controller
from torchrl.contrib.models import ActorCriticNet


class PPOController(Controller):
  def __init__(self, obs_size, action_size,
               gamma=0.99, lmbda=1.0, lr=1e-3,
               alpha=0.5, beta=1.0, clip_ratio=0.2,
               max_grad_norm=1.0, device=None):
    self.device = device

    self.ac_net = ActorCriticNet(obs_size, action_size, 256).to(self.device)
    self.ac_net_optim = torch.optim.Adam(self.ac_net.parameters(), lr=lr)

    self.gamma = gamma
    self.lmbda = lmbda
    self.alpha = alpha
    self.beta = beta
    self.clip_ratio = clip_ratio
    self.max_grad_norm = max_grad_norm

  def act(self, obs):
    with torch.no_grad():
      obs_tensor = torch.Tensor(obs).float().to(self.device)
      _, dist = self.ac_net(obs_tensor)

    action = dist.sample().cpu().numpy()
    return action

  def compute_return(self, obs, action, reward, next_obs, done):  # pylint: disable=unused-argument
    with torch.no_grad():
      values, dist = self.ac_net(obs)
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

      log_probs = dist.log_prob(action).detach()
      values = values[:-1]  # remove the added step to compute returns

      return returns, log_probs, values

  def learn(self, obs, action, reward, next_obs, done,  #pylint: disable=unused-argument
            returns, old_log_probs, advantages):
    values, dist = self.ac_net(obs)

    new_log_probs = dist.log_prob(action)
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                        1 + self.clip_ratio) * advantages
    actor_loss = - torch.min(surr1, surr2).mean()

    critic_loss = torch.nn.functional.mse_loss(values, returns)

    entropy_loss = dist.entropy().mean()

    loss = actor_loss + self.alpha * critic_loss - self.beta * entropy_loss

    self.ac_net_optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.ac_net.parameters(),
                                    self.max_grad_norm)
    self.ac_net_optim.step()

    return dict(
        actor_loss=actor_loss.detach(),
        critic_loss=critic_loss.detach(),
        entropy_loss=entropy_loss.detach()
    )
