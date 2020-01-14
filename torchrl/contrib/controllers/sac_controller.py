import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torchrl.controllers import Controller
from torchrl.contrib.models import QNet, ActorNet


def polyak_average_(source, target, tau=1e-3):
  assert isinstance(source, torch.nn.Module), \
      '"source" should be of type nn.Module, found "{}"'.format(type(source))
  assert isinstance(target, torch.nn.Module), \
      '"target" should be of type nn.Module, found "{}"'.format(type(target))

  for src_param, target_param in zip(source.parameters(), target.parameters()):
    target_param.data.copy_(tau * src_param.data +
                            (1.0 - tau) * target_param.data)


class SACController(Controller):
  def __init__(self, obs_size, action_size, action_lo,
               action_hi, gamma=0.99, tau=5e-3, alpha=0.2,
               lr=3e-4, n_update_interval=1, device=None):
    self.device = device

    self.gamma = gamma
    self.tau = tau
    self.n_update_interval = n_update_interval

    self.qnets = torch.nn.ModuleList([
        QNet(obs_size + action_size, 1, hidden_size=256),
        QNet(obs_size + action_size, 1, hidden_size=256)
    ]).to(self.device)
    self.target_qnets = torch.nn.ModuleList([
        QNet(obs_size + action_size, 1, hidden_size=256),
        QNet(obs_size + action_size, 1, hidden_size=256)
    ]).to(self.device)
    self.target_qnets.load_state_dict(self.qnets.state_dict())

    self.qnet_optim = torch.optim.Adam(self.qnets.parameters(), lr=lr)

    self.actor = ActorNet(obs_size, action_size,
                          hidden_size=256).to(self.device)
    self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)

    self.log_alpha = torch.nn.Parameter(torch.tensor(alpha).log())
    self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr)

    self._steps = 0

  def act(self, obs):
    with torch.no_grad():
      obs_tensor = torch.Tensor(obs).float().to(self.device)

      next_mu, next_log_std = self.actor(obs_tensor)
      policy = Independent(Normal(next_mu, next_log_std.exp()), 1)

      action = policy.sample().tanh()

    return action.cpu().numpy()

  def learn(self, obs, action, reward, next_obs, done):
    with torch.no_grad():
      next_mu, next_log_std = self.actor(next_obs)
      next_policy = Independent(Normal(next_mu, next_log_std.exp()), 1)
      next_action = next_policy.sample()
      next_log_prob = next_policy.log_prob(next_action) - \
                      (1. - next_action.tanh()**2).log().sum(dim=-1)

      q_in = torch.cat([next_obs, next_action], dim=-1)
      q1, q2 = [qnet(q_in) for qnet in self.target_qnets]

      next_q_values = torch.min(q1, q2) - \
                      self.log_alpha.exp() * next_log_prob.unsqueeze(-1)
      expected_q_values = reward + \
                          self.gamma * next_q_values * (1.0 - done.float())

    q_in = torch.cat([obs, action], dim=-1)
    q_values = [qnet(q_in) for qnet in self.qnets]

    q1_loss = F.mse_loss(q_values[0], expected_q_values)
    q2_loss = F.mse_loss(q_values[1], expected_q_values)

    mu, log_std = self.actor(obs)
    policy = Independent(Normal(mu, log_std.exp()), 1)
    action_sample = policy.rsample()

    q_in = torch.cat([obs, action_sample], dim=-1)
    q_values = [qnet(q_in) for qnet in self.qnets]
    log_prob = policy.log_prob(action_sample) -\
               (1. - action_sample.tanh()**2).log().sum(dim=-1)
    policy_loss = - (torch.min(*q_values) -\
                    self.log_alpha.exp().detach() * log_prob.unsqueeze(-1)).mean(dim=0)

    self.qnet_optim.zero_grad()
    (q1_loss + q2_loss).backward()
    self.qnet_optim.step()

    self.actor_optim.zero_grad()
    policy_loss.backward()
    self.actor_optim.step()

    self._steps += 1
    if self._steps % self.n_update_interval == 0:
      polyak_average_(self.qnets, self.target_qnets, self.tau)

    return dict(
        critic1_loss=q1_loss.detach(),
        critic2_loss=q2_loss.detach(),
        policy_loss=policy_loss.detach(),
        alpha=self.log_alpha.exp().detach()
    )
