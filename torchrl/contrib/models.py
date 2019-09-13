import torch
from torch import nn
from torch.distributions import Normal, Categorical


class QNet(nn.Module):
  def __init__(self, input_size, output_size):
    super(QNet, self).__init__()

    self._input_size = input_size
    self._output_size = output_size
    self._hidden_size = 128

    self.net = nn.Sequential(
        nn.Linear(self._input_size, self._hidden_size),
        nn.ReLU(),
        nn.Linear(self._hidden_size, self._hidden_size),
        nn.ReLU(),
        nn.Linear(self._hidden_size, self._output_size)
    )

  def forward(self, obs):
    values = self.net(obs)
    return values


class DDPGActorNet(nn.Module):
  def __init__(self, state_size, action_size, hidden_size):
    super(DDPGActorNet, self).__init__()

    self.state_size = state_size
    self.action_size = action_size
    self.hidden_size = hidden_size

    self._weight_init = 3e-3

    self.net = nn.Sequential(
        nn.Linear(self.state_size, self.hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, self.hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, self.action_size),
        nn.Tanh()
    )

    self._init_weights()

  def forward(self, obs):
    return self.net(obs)

  def _init_weights(self):
    nn.init.uniform_(self.net[-2].weight, -self._weight_init, self._weight_init)
    nn.init.uniform_(self.net[-2].bias, -self._weight_init, self._weight_init)


class DDPGCriticNet(nn.Module):
  def __init__(self, state_size, action_size, hidden_size):
    super(DDPGCriticNet, self).__init__()

    self.state_size = state_size
    self.action_size = action_size
    self.hidden_size = hidden_size

    self._weight_init = 3e-3

    self.net = nn.Sequential(
        nn.Linear(self.state_size + self.action_size, self.hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, self.hidden_size),
        nn.ReLU(),
        nn.Linear(self.hidden_size, 1)
    )

    self._init_weights()

  def forward(self, obs, action):
    return self.net(torch.cat([obs, action], dim=1))

  def _init_weights(self):
    nn.init.uniform_(self.net[-1].weight, -self._weight_init, self._weight_init)
    nn.init.uniform_(self.net[-1].bias, -self._weight_init, self._weight_init)


class A2CNet(nn.Module):
  def __init__(self, input_size, output_size, hidden_size):
    super(A2CNet, self).__init__()

    self._input_size = input_size
    self._output_size = output_size
    self._hidden_size = hidden_size

    self.critic = nn.Sequential(
        nn.Linear(self._input_size, self._hidden_size),
        nn.ReLU(),
        nn.Linear(self._hidden_size, 1)
    )

    self.actor = nn.Sequential(
        nn.Linear(self._input_size, self._hidden_size),
        nn.ReLU(),
        nn.Linear(self._hidden_size, self._output_size),
        nn.Softmax(dim=1)
    )

  def forward(self, obs):
    value = self.critic(obs)
    policy = self.actor(obs)
    dist = Categorical(policy)
    return value, dist


class ActorCriticNet(nn.Module):
  def __init__(self, input_size, output_size, hidden_size, std=0.0):
    super(ActorCriticNet, self).__init__()

    self._input_size = input_size
    self._output_size = output_size
    self._hidden_size = hidden_size

    self.critic = nn.Sequential(
        nn.Linear(self._input_size, self._hidden_size),
        nn.ReLU(),
        nn.Linear(self._hidden_size, 1)
    )

    self.actor = nn.Sequential(
        nn.Linear(self._input_size, self._hidden_size),
        nn.ReLU(),
        nn.Linear(self._hidden_size, self._output_size)
    )

    self.log_std = nn.Parameter(torch.ones(1, self._output_size) * std)

    self.apply(self.init_weights)

  def forward(self, obs):
    value = self.critic(obs)
    mean = self.actor(obs)
    std = self.log_std.exp().expand_as(mean)
    dist = Normal(mean, std)
    return value, dist

  @staticmethod
  def init_weights(module):
    if isinstance(module, nn.Linear):
      nn.init.normal_(module.weight, mean=0., std=0.1)
      nn.init.constant_(module.bias, 0.1)
