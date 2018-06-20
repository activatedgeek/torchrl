import os
from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

from torchrl.learners import BaseLearner
from torchrl.policies import OUNoise
from torchrl.models import DDPGActorNet, DDPGCriticNet


def polyak_average_(source, target, tau=1e-3):
  """
  In-place Polyak Average from the source to the target
  :param tau: Polyak Averaging Parameter
  :param source: Source Module
  :param target: Target Module
  :return:
  """
  assert isinstance(source, nn.Module), \
      '"source" should be of type nn.Module, found "{}"'.format(type(source))
  assert isinstance(target, nn.Module), \
      '"target" should be of type nn.Module, found "{}"'.format(type(target))

  for src_param, target_param in zip(source.parameters(), target.parameters()):
    target_param.data.copy_(tau * src_param.data +
                            (1.0 - tau) * target_param.data)


class BaseDDPGLearner(BaseLearner):
  def __init__(self, observation_space, action_space,
               actor_lr=1e-4,
               critic_lr=1e-3,
               gamma=0.99,
               tau=1e-2):
    super(BaseDDPGLearner, self).__init__(observation_space, action_space)

    self.actor = DDPGActorNet(observation_space.shape[0],
                              action_space.shape[0], 256)
    self.target_actor = deepcopy(self.actor)
    self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

    self.critic = DDPGCriticNet(observation_space.shape[0],
                                action_space.shape[0], 256)
    self.target_critic = deepcopy(self.critic)
    self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

    self.gamma = gamma
    self.tau = tau
    self.noise = OUNoise(self.action_space)

    self.train()

    # Internal vars
    self._step = 0

  def act(self, obs, **kwargs):
    action = self.actor(obs)
    action = action.cpu().numpy()
    action = self.noise.get_action(action, self._step)
    action = self.clip_action(action)

    self._step += 1

    return np.expand_dims(action, axis=1)

  def clip_action(self, action: np.array):
    low_bound = self.action_space.low
    upper_bound = self.action_space.high

    action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
    action = np.clip(action, low_bound, upper_bound)

    return action

  def learn(self, obs, action, reward, next_obs, done, **kwargs):
    actor_loss = - self.critic(obs, self.actor(obs)).mean()

    next_action = self.target_actor(next_obs).detach()
    current_q = self.critic(obs, action)
    target_q = reward + (1.0 - done.float()) * self.gamma * self.target_critic(next_obs, next_action)  # pylint: disable=line-too-long

    critic_loss = F.mse_loss(current_q, target_q.detach())

    self.actor_optim.zero_grad()
    actor_loss.backward()
    self.actor_optim.step()

    self.critic_optim.zero_grad()
    critic_loss.backward()
    self.critic_optim.step()

    polyak_average_(self.actor, self.target_actor, self.tau)
    polyak_average_(self.critic, self.target_critic, self.tau)

    return actor_loss.detach().cpu().item(), \
        critic_loss.detach().cpu().item()

  def reset(self):
    self.noise.reset()
    self._step = 0

  def cuda(self):
    self.actor.cuda()
    self.target_actor.cuda()
    self.critic.cuda()
    self.target_critic.cuda()
    self.is_cuda = True

  def train(self):
    self.actor.train()
    self.target_actor.train()
    self.critic.train()
    self.target_critic.train()
    self.training = True

  def eval(self):
    self.actor.eval()
    self.target_actor.eval()
    self.critic.eval()
    self.target_critic.eval()
    self.training = False

  def save(self, save_dir):
    actor_file_name = os.path.join(save_dir, 'actor.pth')
    torch.save(self.actor.state_dict(), actor_file_name)

    critic_file_name = os.path.join(save_dir, 'critic.pth')
    torch.save(self.critic.state_dict(), critic_file_name)

  def load(self, load_dir):
    actor_file_name = os.path.join(load_dir, 'actor.pth')
    self.actor.load_state_dict(torch.load(actor_file_name))
    self.target_actor = deepcopy(self.actor)

    critic_file_name = os.path.join(load_dir, 'critic.pth')
    self.critic.load_state_dict(torch.load(critic_file_name))
    self.target_critic = deepcopy(self.critic)
