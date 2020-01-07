import torch
import numpy as np
from torchrl.controllers import Controller
from torchrl.contrib.models import DDPGActorNet, DDPGCriticNet


class OUNoise:
  def __init__(self, action_dim, mu=0.0, theta=0.15,
               max_sigma=0.3, min_sigma=0.3, decay_period=100000):
    self.mu = mu
    self.theta = theta
    self.sigma = max_sigma
    self.max_sigma = max_sigma
    self.min_sigma = min_sigma
    self.decay_period = decay_period
    self.action_dim = action_dim

    self.state = None
    self.step = 0
    self.reset()

  def reset(self):
    self.state = np.ones(self.action_dim) * self.mu
    self.step = 0

  def evolve_state(self):
    x = self.state
    dx = self.theta * (self.mu - x) + \
         self.sigma * np.random.randn(self.action_dim)
    self.state = x + dx
    return self.state

  def get_action(self, action):
    ou_state = self.evolve_state()
    self.sigma = self.max_sigma - \
                 (self.max_sigma - self.min_sigma) * \
                 min(1.0, self.step / self.decay_period)
    self.step += 1
    return action + ou_state


def polyak_average_(source, target, tau=1e-3):
  """
  In-place Polyak Average from the source to the target
  :param tau: Polyak Averaging Parameter
  :param source: Source Module
  :param target: Target Module
  :return:
  """
  assert isinstance(source, torch.nn.Module), \
      '"source" should be of type nn.Module, found "{}"'.format(type(source))
  assert isinstance(target, torch.nn.Module), \
      '"target" should be of type nn.Module, found "{}"'.format(type(target))

  for src_param, target_param in zip(source.parameters(), target.parameters()):
    target_param.data.copy_(tau * src_param.data +
                            (1.0 - tau) * target_param.data)


class DDPGController(Controller):
  def __init__(self, obs_size, action_size,
               action_low, action_high,
               actor_lr=1e-4, critic_lr=1e-3, gamma=0.99,
               tau=1e-2, n_reset_interval=100, device=None):
    self.device = device
    self.action_low = action_low
    self.action_high = action_high

    self.actor = DDPGActorNet(obs_size, action_size, 256).to(self.device)
    self.target_actor = DDPGActorNet(obs_size, action_size, 256).to(self.device)
    self.target_actor.load_state_dict(self.actor.state_dict())

    self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

    self.critic = DDPGCriticNet(obs_size, action_size, 256).to(self.device)
    self.target_critic = DDPGCriticNet(obs_size, action_size,
                                       256).to(self.device)
    self.target_critic.load_state_dict(self.critic.state_dict())

    self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    self.gamma = gamma
    self.tau = tau
    self.n_reset_interval = n_reset_interval
    self.noise = OUNoise(action_size)

  def act(self, obs):
    with torch.no_grad():
      obs_tensor = torch.Tensor(obs).float().to(self.device)
      action = self.actor(obs_tensor)

    action = action.cpu().detach().numpy()
    action = self.noise.get_action(action)
    action = np.clip(action, self.action_low, self.action_high)

    if self.noise.step % self.n_reset_interval == 0:
      self.noise.reset()

    return action

  def learn(self, obs, action, reward, next_obs, done):
    self.actor.train()
    self.critic.train()

    actor_loss = - self.critic(obs, self.actor(obs)).mean()

    current_q = self.critic(obs, action)
    with torch.no_grad():
      next_action = self.target_actor(next_obs)
      target_q = reward + (1.0 - done.float()) * self.gamma * self.target_critic(next_obs, next_action)  # pylint: disable=line-too-long

    critic_loss = torch.nn.functional.mse_loss(current_q, target_q)

    self.actor_optim.zero_grad()
    actor_loss.backward()
    self.actor_optim.step()

    self.critic_optim.zero_grad()
    critic_loss.backward()
    self.critic_optim.step()

    polyak_average_(self.actor, self.target_actor, self.tau)
    polyak_average_(self.critic, self.target_critic, self.tau)

    return dict(
        actor_loss=actor_loss.detach().cpu(),
        critic_loss=critic_loss.detach().cpu()
    )
