from copy import deepcopy
import torch
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable

from torchrl import BaseLearner
from torchrl.utils import polyak_average

from models import Actor, Critic


class BaseDDPGLearner(BaseLearner):
    def __init__(self, observation_space, action_space, noise,
                 actor_lr=1e-4,
                 critic_lr=1e-3,
                 gamma=0.99,
                 tau=1e-2):
        super(BaseDDPGLearner, self).__init__(observation_space, action_space)

        self.actor = Actor(observation_space.shape[0], action_space.shape[0])
        self.target_actor = deepcopy(self.actor)
        self.actor_optim = Adam(self.actor.parameters(), lr=actor_lr)

        self.critic = Critic(observation_space.shape[0], action_space.shape[0])
        self.target_critic = deepcopy(self.critic)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma
        self.tau = tau
        self.noise = noise

        self.train()

    def act(self, obs, **kwargs):
        action = self.actor(obs)
        action = action.cpu().data.numpy()
        action += self.noise()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return np.expand_dims(action, axis=1)

    def learn(self, obs, action, reward, next_obs, done, **kwargs):
        obs_tensor = Variable(torch.from_numpy(obs).float())
        action_tensor = Variable(torch.from_numpy(action).float())
        reward_tensor = Variable(torch.from_numpy(reward).float())
        next_obs_tensor = Variable(torch.from_numpy(next_obs).float())

        if self.is_cuda:
            obs_tensor = obs_tensor.cuda()
            action_tensor = action_tensor.cuda()
            reward_tensor = reward_tensor.cuda()
            next_obs_tensor = next_obs_tensor.cuda()

        current_q = self.critic(obs_tensor, action_tensor)

        target_q = reward_tensor + self.gamma * self.target_critic(next_obs_tensor, self.target_actor(next_obs_tensor))

        critic_loss = F.mse_loss(current_q, target_q)

        critic_loss.backward()
        self.critic_optim.step()
        self.critic_optim.zero_grad()

        actor_loss = - self.critic(obs_tensor, self.actor(obs_tensor)).mean()

        actor_loss.backward()
        self.actor_optim.step()
        self.actor_optim.zero_grad()

        polyak_average(self.actor, self.target_actor, self.tau)
        polyak_average(self.critic, self.target_critic, self.tau)

        return actor_loss.detach().cpu().data.numpy(), critic_loss.detach().cpu().data.numpy()

    def reset(self):
        self.noise.reset()

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
