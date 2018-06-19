import numpy as np
import torch
from torch.optim import Adam

from torchrl.learners import BaseLearner
from torchrl.models import ActorCriticNet


class BasePPOLearner(BaseLearner):
  def __init__(self, observation_space, action_space,
               lr=1e-3,
               gamma=0.99,
               lmbda=0.01,
               alpha=0.5,
               beta=1.0,
               clip_ratio=0.2,
               max_grad_norm=1.0):
    super(BasePPOLearner, self).__init__(observation_space, action_space)

    self.ac_net = ActorCriticNet(observation_space.shape[0],
                                 action_space.shape[0], 256)
    self.ac_net_optim = Adam(self.ac_net.parameters(), lr=lr)

    self.gamma = gamma
    self.lmbda = lmbda
    self.alpha = alpha
    self.beta = beta
    self.clip_ratio = clip_ratio
    self.max_grad_norm = max_grad_norm

    self.train()

  def act(self, obs):
    _, dist = self.ac_net(obs)
    action = dist.sample()
    return action.unsqueeze(1).cpu().data.numpy()

  def compute_returns(self, obs, action, reward, next_obs, done):  # pylint: disable=unused-argument
    with torch.no_grad():
      obs_tensor = torch.from_numpy(obs).float()
      action_tensor = torch.from_numpy(action).float()

      if self.is_cuda:
        obs_tensor = obs_tensor.cuda()
        action_tensor = torch.from_numpy(action).float()

      values, dist = self.ac_net(obs_tensor)
      values = values.cpu().data.numpy()
      if not done[-1]:
        next_obs_tensor = torch.from_numpy(next_obs[-1]).float().unsqueeze(0)
        if self.is_cuda:
          next_obs_tensor = next_obs_tensor.cuda()

        next_value, _ = self.ac_net(next_obs_tensor)
        next_value = next_value.cpu().data.numpy()
        values = np.append(values, next_value, axis=0)
      else:
        values = np.append(values, np.array([[0.0]]), axis=0)

      returns = [0.0] * len(reward)
      gae = 0.0
      for step in reversed(range(len(reward))):
        delta = reward[step] + self.gamma * values[step + 1] - values[step]
        gae = delta + self.gamma * self.lmbda * gae
        returns[step] = gae + values[step]

      returns = np.array(returns)

      log_probs = dist.log_prob(action_tensor).detach()
      values = values[:-1]  # remove the added step to compute returns

      return returns, log_probs, values

  def learn(self, obs, action, reward, next_obs, done,  #pylint: disable=unused-argument
            returns, old_log_probs, advantages):
    obs_tensor = torch.from_numpy(obs).float()
    action_tensor = torch.from_numpy(action).float()
    return_tensor = torch.from_numpy(returns).float()
    advantage_tensor = torch.from_numpy(advantages).float()

    if self.is_cuda:
      obs_tensor = obs_tensor.cuda()
      action_tensor = action_tensor.cuda()
      return_tensor = return_tensor.cuda()
      old_log_probs = old_log_probs.cuda()
      advantage_tensor = advantage_tensor.cuda()

    values, dist = self.ac_net(obs_tensor)

    new_log_probs = dist.log_prob(action_tensor)
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantage_tensor
    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                        1 + self.clip_ratio) * advantage_tensor
    actor_loss = - torch.min(surr1, surr2).mean()

    critic_loss = (return_tensor - values).pow(2).mean()

    entropy_loss = dist.entropy().mean()

    loss = actor_loss + self.alpha * critic_loss - self.beta * entropy_loss

    self.ac_net_optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(self.ac_net.parameters(),
                                    self.max_grad_norm)
    self.ac_net_optim.step()

    return actor_loss.detach().cpu().data.numpy(), \
        critic_loss.detach().cpu().data.numpy(), \
        entropy_loss.detach().cpu().data.numpy()

  def cuda(self):
    self.ac_net.cuda()
    self.is_cuda = True

  def train(self):
    self.ac_net.train()
    self.training = True

  def eval(self):
    self.ac_net.eval()
    self.training = False
