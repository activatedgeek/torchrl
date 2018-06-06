import numpy as np
import torch
from torch.optim import Adam
from torch.autograd import Variable
from torch.distributions import Normal

from torchrl.learners import BaseLearner
from torchrl.models import ActorCriticNet


class BasePPOLearner(BaseLearner):
  def __init__(self, observation_space, action_space,
               lr=1e-3,
               gamma=0.99,
               lmbda=0.01,
               alpha=0.5,
               beta=1.0,
               clip_ratio=0.2):
    super(BasePPOLearner, self).__init__(observation_space, action_space)

    self.ac_net = ActorCriticNet(observation_space.shape[0],
                                 action_space.shape[0], 256)
    self.ac_net_optim = Adam(self.ac_net.parameters(), lr=lr)

    self.gamma = gamma
    self.lmbda = lmbda
    self.alpha = alpha
    self.beta = beta
    self.clip_ratio = clip_ratio

    self.train()

  def act(self, obs):
    _, mean, std = self.ac_net(obs)
    dist = Normal(mean, std)
    action = dist.sample()
    return action.unsqueeze(1).cpu().data.numpy()

  def compute_returns(self, obs, action, reward, next_obs, done):  # pylint: disable=unused-argument
    obs_tensor = Variable(torch.from_numpy(obs).float(), volatile=True)
    if self.is_cuda:
      obs_tensor = obs_tensor.cuda()

    values, _, _ = self.ac_net(obs_tensor)
    values = values.cpu().data.numpy()
    if not done[-1]:
      next_obs_tensor = Variable(torch.from_numpy(
          next_obs[-1]).float().unsqueeze(0), volatile=True)
      if self.is_cuda:
        next_obs_tensor = next_obs_tensor.cuda()

      next_value, _, _ = self.ac_net(next_obs_tensor)
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
    returns = returns[::-1]

    return returns

  def compute_old_log_probs(self, obs, action):
    obs_tensor = Variable(torch.from_numpy(obs).float(), volatile=True)
    if self.is_cuda:
      obs_tensor = obs_tensor.cuda()

    _, mean, std = self.ac_net(obs_tensor)
    dist = Normal(mean, std)
    return dist.log_prob(action)

  def learn(self, obs, action, reward, next_obs, done, returns):  # pylint: disable=unused-argument
    obs_tensor = Variable(torch.from_numpy(obs).float())
    action_tensor = Variable(torch.from_numpy(action).long())
    return_tensor = Variable(torch.from_numpy(returns).float())

    if self.is_cuda:
      obs_tensor = obs_tensor.cuda()
      action_tensor = action_tensor.cuda()
      return_tensor = return_tensor.cuda()

    values, mean, std = self.ac_net(obs_tensor)
    dist = Normal(mean, std)

    advantages = return_tensor - values

    # TODO: compute log probs for new and old
    ratio = (new_log_probs - old_log_probs).exp()
    surr1 = ratio * advantages.detach()
    surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio,
                        1 + self.clip_ratio) * advantages.detach()
    actor_loss = - torch.min(surr1, surr2).mean()

    critic_loss = advantages.pow(2).mean()

    # TODO: compute current distribution entropy
    entropy_loss = - (prob * prob.log()).sum(dim=1).mean()

    loss = actor_loss + self.alpha * critic_loss + self.beta * entropy_loss

    self.ac_net_optim.zero_grad()
    loss.backward()
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
