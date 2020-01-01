import abc
import torch
import numpy as np
from torchrl.utils.multi_envs import get_gym_spaces
from torchrl.problems import base_hparams
from torchrl.contrib.models import A2CNet
from torchrl.problems import GymProblem


class BaseAgent(metaclass=abc.ABCMeta):
  """
  This is base agent specification which can encapsulate everything
  how a Reinforcement Learning Algorithm would function.
  """
  def __init__(self, observation_space, action_space):
    self.observation_space = observation_space
    self.action_space = action_space
    self.device = None

  @property
  def models(self) -> list:
    """
    This routine must return the list of trainable
    networks which external routines might want to
    generally operate on
    :return:
    """
    raise NotImplementedError

  @property
  def checkpoint(self) -> object:
    """
    This method must return an arbitrary object
    which defines the complete state of the agent
    to restore at any point in time
    """
    raise NotImplementedError

  @checkpoint.setter
  def checkpoint(self, cp):
    """
    This method must be the complement of
    `self.state` and restore the state
    """
    raise NotImplementedError

  @abc.abstractmethod
  def act(self, *args, **kwargs):
    """
    This is the method that should be called at every step of the episode.
    IMPORTANT: This method should be compatible with batches

    :param state: Representation of the state
    :return: identity of the action to be taken, as desired by the environment
    """
    raise NotImplementedError

  @abc.abstractmethod
  def learn(self, *args, **kwargs) -> dict:
    """
    This method represents the learning step
    """
    raise NotImplementedError

  def reset(self):
    """
    Optional function to reset learner's internals
    :return:
    """
    # pass

  def to(self, device: torch.device):
    """
    This routine is takes the agent's :code:`models` attribute
    and sends them to a device.

    See https://pytorch.org/docs/stable/nn.html#torch.nn.Module.to.

    Args:
        device (:class:`torch.device`):

    Returns:
        Updated class reference.
    """
    self.device = device
    for model in self.models:
      model.to(device)

    return self

  def train(self, flag: bool = True):
    """
    This routine is takes the agent's :code:`models` attribute
    and applies the training flag.

    See https://pytorch.org/docs/stable/nn.html#torch.nn.Module.train.

    Args:
        flag (bool): :code:`True` or :code:`False`
    """
    for model in self.models:
      model.train(flag)

  def obs_to_tensor(self, obs):
    with torch.no_grad():
      batch_obs_tensor = torch.from_numpy(
          np.array(obs)
      ).float()

    if self.device:
      batch_obs_tensor = batch_obs_tensor.to(self.device)

    return batch_obs_tensor

class BaseA2CAgent(BaseAgent):
  def __init__(self, observation_space, action_space,
               lr=1e-3,
               gamma=0.99,
               lmbda=1.0,
               alpha=0.5,
               beta=1.0):
    super(BaseA2CAgent, self).__init__(observation_space, action_space)

    self.ac_net = A2CNet(observation_space.shape[0], action_space.n, 256)
    self.ac_net_optim = torch.optim.Adam(self.ac_net.parameters(), lr=lr)

    self.gamma = gamma
    self.lmbda = lmbda
    self.alpha = alpha
    self.beta = beta

  @property
  def models(self):
    return [self.ac_net]

  @property
  def checkpoint(self):
    return {
        'ac_net': self.ac_net.state_dict()
    }

  @checkpoint.setter
  def checkpoint(self, cp):
    self.ac_net.load_state_dict(cp['ac_net'])

  def act(self, obs):
    obs_tensor = self.obs_to_tensor(obs)

    _, dist = self.ac_net(obs_tensor)
    action = dist.sample()
    return action.unsqueeze(1).cpu().numpy()

  def compute_returns(self, obs, action, reward, next_obs, done):  # pylint: disable=unused-argument
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

    return actor_loss.detach().cpu().item(), \
        critic_loss.detach().cpu().item(), \
        entropy_loss.detach().cpu().item()


class A2CProblem(GymProblem):
  def train(self, history_list: list):
    history_list = self.hist_to_tensor(history_list, device=self.device)

    batch_history = self.merge_histories(*history_list)
    returns = torch.cat([
        self.agent.compute_returns(*history)
        for history in history_list
    ], dim=0)

    actor_loss, critic_loss, entropy_loss = self.agent.learn(*batch_history,
                                                             returns)
    return {'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss}


class A2CCartpole(A2CProblem):
  def init_agent(self):
    observation_space, action_space = get_gym_spaces(self.runner.make_env)

    agent = BaseA2CAgent(
        observation_space,
        action_space,
        lr=self.hparams.actor_lr,
        gamma=self.hparams.gamma,
        lmbda=self.hparams.lmbda,
        alpha=self.hparams.alpha,
        beta=self.hparams.beta)

    return agent

  @staticmethod
  def hparams_a2c_cartpole():
    params = base_hparams.base_pg()

    params.env_id = 'CartPole-v0'

    params.num_processes = 16

    params.rollout_steps = 5
    params.max_episode_steps = 500
    params.num_total_steps = int(1e6)

    params.alpha = 0.5
    params.gamma = 0.99
    params.beta = 1e-3
    params.lmbda = 1.0

    params.batch_size = 128
    params.tau = 1e-2
    params.actor_lr = 3e-4

    return params
