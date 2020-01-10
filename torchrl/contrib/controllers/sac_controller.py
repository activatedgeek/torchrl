import torch
import torch.nn.functional as F
from torch.distributions import Normal, Independent
from torchrl.controllers import Controller
from torchrl.contrib.models import QNet, ActorNet


class SACController(Controller):
  def __init__(self, obs_size, action_size, gamma=0.99,
               tau=5e-3, alpha=1e-2, lr=3e-4, device=None):
    self.device = device

    self.gamma = gamma
    self.tau = tau

    self.qnets = torch.nn.ModuleList([
        QNet(obs_size + action_size, 1, hidden_size=256),
        QNet(obs_size + action_size, 1, hidden_size=256)
    ]).to(self.device)
    self.qnets[1].load_state_dict(self.qnets[0].state_dict())

    self.qnet_optim = torch.optim.Adam(self.qnets.parameters(), lr=lr)

    self.actor = ActorNet(obs_size, action_size,
                          hidden_size=256).to(self.device)
    self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=lr)

    self.alpha = torch.nn.Parameter(torch.tensor(alpha))
    self.alpha_optim = torch.optim.Adam([self.alpha], lr=lr)

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
      policy = Independent(Normal(next_mu, next_log_std.exp()), 1)
      next_action = policy.sample()

      q_in = torch.cat([next_obs, next_action], dim=-1)
      q1, q2 = [qnet(q_in) for qnet in self.qnets]

      next_log_prob = policy.log_prob(next_action) - \
                      (1. - next_action.tanh()**2).log().sum(dim=-1)

      next_q_values = torch.min(q1, q2) - \
                      self.alpha * next_log_prob.unsqueeze(-1)
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
                     self.alpha.detach() * log_prob.unsqueeze(-1)).mean(dim=0)


    self.qnet_optim.zero_grad()
    (q1_loss + q2_loss).backward()
    self.qnet_optim.step()

    self.actor_optim.zero_grad()
    policy_loss.backward()
    self.actor_optim.step()

    return dict(
        critic1_loss=q1_loss.detach(),
        critic2_loss=q2_loss.detach(),
        policy_loss=policy_loss.detach(),
        alpha=self.alpha.detach()
    )
