import abc
import torch
import numpy as np
from torchrl.utils.multi_envs import get_gym_spaces
from torchrl.problems import base_hparams
from torchrl.contrib.models import ActorCriticNet
from torchrl.problems import GymProblem
from torchrl.utils import minibatch_generator


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

class BasePPOAgent(BaseAgent):
  def __init__(self, observation_space, action_space,
               lr=1e-3,
               gamma=0.99,
               lmbda=0.01,
               alpha=0.5,
               beta=1.0,
               clip_ratio=0.2,
               max_grad_norm=1.0):
    super(BasePPOAgent, self).__init__(observation_space, action_space)

    self.ac_net = ActorCriticNet(observation_space.shape[0],
                                 action_space.shape[0], 256)
    self.ac_net_optim = torch.optim.Adam(self.ac_net.parameters(), lr=lr)

    self.gamma = gamma
    self.lmbda = lmbda
    self.alpha = alpha
    self.beta = beta
    self.clip_ratio = clip_ratio
    self.max_grad_norm = max_grad_norm

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

    return actor_loss.detach().cpu().item(), \
        critic_loss.detach().cpu().item(), \
        entropy_loss.detach().cpu().item()


class PPOProblem(GymProblem):
  def train(self, history_list: list):
    history_list = self.hist_to_tensor(history_list, device=self.device)

    batch_history = self.merge_histories(*history_list)
    data = [self.agent.compute_returns(*history) for history in history_list]
    returns, log_probs, values = self.merge_histories(*data)
    advantages = returns - values

    # Train the agent
    actor_loss, critic_loss, entropy_loss = None, None, None
    for _ in range(self.hparams.ppo_epochs):
      for data in minibatch_generator(*batch_history,
                                      returns, log_probs, advantages,
                                      minibatch_size=self.hparams.batch_size):
        actor_loss, critic_loss, entropy_loss = self.agent.learn(*data)

    return {'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'entropy_loss': entropy_loss}


class PPOPendulum(PPOProblem):
  def init_agent(self):
    observation_space, action_space = get_gym_spaces(self.runner.make_env)

    agent = BasePPOAgent(
        observation_space,
        action_space,
        lr=self.hparams.actor_lr,
        gamma=self.hparams.gamma,
        lmbda=self.hparams.lmbda,
        alpha=self.hparams.alpha,
        beta=self.hparams.beta,
        max_grad_norm=self.hparams.max_grad_norm)

    return agent

  @staticmethod
  def hparams_ppo_pendulum():
    params = base_hparams.base_ppo()

    params.env_id = 'Pendulum-v0'

    params.rollout_steps = 20
    params.num_processes = 16
    params.num_total_steps = int(5e6)

    params.batch_size = 64

    params.actor_lr = 3e-4

    params.alpha = 0.5
    params.gamma = 0.99
    params.beta = 1e-3
    params.lmbda = 0.95

    params.clip_ratio = 0.2
    params.max_grad_norm = 1.0
    params.ppo_epochs = 4

    return params
