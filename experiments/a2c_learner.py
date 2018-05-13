import torch
from torch.optim import RMSprop
from torch.autograd import Variable
from torch.distributions import Categorical

from torchrl import BaseLearner

from models import ACNet


class BaseA2CLearner(BaseLearner):
    def __init__(self, observation_space, action_space,
                 lr=1e-3,
                 gamma=0.99,
                 clip_grad_norm=10.0):
        super(BaseA2CLearner, self).__init__(observation_space, action_space)

        self.ac_net = ACNet(observation_space.shape[0], action_space.n)
        self.ac_net_optim = RMSprop(self.ac_net.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_grad_norm = clip_grad_norm

        self.train()

    def act(self, obs, **kwargs):
        _, prob = self.ac_net(obs)
        dist = Categorical(prob)
        action = dist.sample()
        return action.unsqueeze(1).cpu().data.numpy()

    # @TODO: accomodate for episode boundaries when batching
    def learn(self, obs, action, reward, next_obs, done, **kwargs):
        obs_tensor = Variable(torch.from_numpy(obs).float(), requires_grad=True)
        action_tensor = Variable(torch.from_numpy(action).float(), requires_grad=True)
        reward_tensor = Variable(torch.from_numpy(reward).float())
        next_obs_tensor = Variable(torch.from_numpy(next_obs).float(), volatile=True)

        if self.is_cuda:
            obs_tensor = obs_tensor.cuda()
            action_tensor = action_tensor.cuda()
            reward_tensor = reward_tensor.cuda()
            next_obs_tensor = next_obs_tensor.cuda()

        expected_return = Variable(torch.zeros(1, 1))
        if not done[-1]:
            expected_return, _ = self.ac_net(next_obs_tensor[-1].unsqueeze(0))

        value_batch.append(expected_return)

        policy_loss = 0.0
        value_loss = 0.0
        gae = torch.zeros(1, 1)
        for i in reversed(range(len(reward_batch))):
            expected_return = reward_batch[i] + self.gamma * expected_return
            advantage = expected_return - value_batch[i]

            value_loss += advantage.pow(2)

            td_error = reward_batch[i] + self.gamma * value_batch[i + 1].data - value_batch[i].data
            gae = td_error + self.gamma * self.tau * gae

            policy_loss -= log_prob_batch[i] * Variable(gae) + self.beta * entropy_batch[i]

        (policy_loss + value_loss).backward()
        torch.nn.utils.clip_grad_norm(self.ac_net.parameters(), self.clip_grad_norm)

        self.ac_net_optim.step()
        self.ac_net_optim.zero_grad()
        self._clear()

    def cuda(self):
        self.ac_net.cuda()
        self.is_cuda = True

    def train(self):
        self.ac_net.train()
        self.training = True

    def eval(self):
        self.ac_net.eval()
        self.training = False
