import time
import gym
import numpy as np
import torch
import functools
from torch.autograd import Variable

from torchrl.learners import BaseLearner
from torchrl.multi_proc_wrapper import MultiProcWrapper


DEFAULT_MAX_STEPS = int(1e6)


class EpisodeRunner:
    """
    EpisodeRunner is a utility wrapper to run episodes on a single Gym-like environment
    object. Each call to the `run()` method will run one episode for specified
    number of steps. Call `reset()` to reuse the same object again
    """
    def __init__(self, env: gym.Env, max_steps: int = DEFAULT_MAX_STEPS):
        """
        :param env: Environment with Gym-like API
        :param max_steps: Maximum number of steps per episode (useful for non-episodic environments)
        """
        self.env = env

        self.max_steps = max_steps

        self._obs = None
        self._done = False

        # Stats
        self._rollout_duration = 0.0

        self.reset()

    def reset(self):
        """
        Reset internal state for the `run` method to be reused
        """
        self._obs = self.env.reset()
        self._done = False

        self._rollout_duration = 0.0

    def is_done(self):
        """
        :return: True if the episode has ended, False otherwise
        """
        return self._done

    def act(self, learner):
        """
        This routine is called from the `run` routine every time an action
        is needed for the environment to step.
        """
        obs_tensor = torch.from_numpy(self._obs).unsqueeze(0).float()
        obs_tensor = Variable(obs_tensor, volatile=True)
        if learner.is_cuda:
            obs_tensor = obs_tensor.cuda()

        action = learner.act(obs_tensor)
        # `act` is a batch call but, for a single episode run this is always one action
        return action[0][0]

    def collect(self, learner: BaseLearner, steps: int = None, render: bool = False, fps: int = 30, store: bool = False):
        """

        :param learner: An agent of type BaseLearner
        :param steps: Number of maximum steps in the current rollout
        :param render: Flag to render the environment, True or False
        :param fps: Rendering rate of the environment, frames per second if render is True
        :param store: Flag to store the history of the run
        :return: batch of transitions
        """
        assert not self._done, 'EpisodeRunner has ended. Call .reset() to reuse.'

        steps = steps or self.max_steps

        if render:
            self.env.render()
            time.sleep(1. / fps)

        is_discrete = self.env.action_space.__class__.__name__ == 'Discrete'

        obs_history, action_history, reward_history, next_obs_history, done_history = \
            self.init_run_history(self.env.observation_space, self.env.action_space)

        rollout_start = time.time()

        while not self._done and steps:
            action = self.act(learner)
            next_obs, reward, self._done, _ = self.env.step(action)

            if store:
                obs_history = np.append(obs_history, np.expand_dims(self._obs, axis=0), axis=0)
                action = np.array([[action]]) if is_discrete else np.expand_dims(action, axis=0)
                action_history = np.append(action_history, action, axis=0)
                reward_history = np.append(reward_history, np.array([[reward]]), axis=0)
                next_obs_history = np.append(next_obs_history, np.expand_dims(next_obs, axis=0), axis=0)
                done_history = np.append(done_history, np.array([[int(self._done)]]), axis=0)

            self._obs = next_obs
            steps -= 1

            if render:
                self.env.render()
                time.sleep(1. / fps)

        self._rollout_duration = time.time() - rollout_start

        if store:
            return obs_history, action_history, reward_history, next_obs_history, done_history

    def stop(self):
        self.env.close()

    def get_stats(self) -> dict:
        return {
            'duration': self._rollout_duration,
        }

    @staticmethod
    def init_run_history(observation_space: gym.Space, action_space: gym.Space):
        is_discrete = action_space.__class__.__name__ == 'Discrete'

        obs_history = np.empty((0, *observation_space.shape), dtype=np.float)
        action_history = np.empty((0, *((1,) if is_discrete else action_space.shape)),
                                  dtype=np.int if is_discrete else np.float)
        reward_history = np.empty((0, 1), dtype=np.float)
        next_obs_history = np.empty_like(obs_history)
        done_history = np.empty((0, 1), dtype=np.int)

        return obs_history, action_history, reward_history, next_obs_history, done_history

    @staticmethod
    def merge_histories(observation_space: gym.Space, action_space: gym.Space, *sources: tuple) -> tuple:
        target = EpisodeRunner.init_run_history(observation_space, action_space)
        return tuple([np.concatenate((tgt, *src), axis=0) for tgt, *src in zip(target, *sources)])


def make_runner(env_id: str, seed: int = None, max_steps: int = DEFAULT_MAX_STEPS):
    env = gym.make(env_id)
    if seed is not None:
        env.seed(seed)
    return EpisodeRunner(env, max_steps=max_steps)


class MultiEpisodeRunner(MultiProcWrapper):
    """
    This class is the parallel version of EpisodeRunner
    """

    def __init__(self, env_id: str, max_steps: int = DEFAULT_MAX_STEPS, n_runners=2, base_seed: int = 0,
                 daemon=True, autostart=True):
        obj_fns =[
            functools.partial(make_runner, env_id,
                              None if base_seed is None else base_seed + rank, max_steps=max_steps)
            for rank in range(1, n_runners + 1)
        ]
        super(MultiEpisodeRunner, self).__init__(obj_fns, daemon=daemon, autostart=autostart)

    def reset(self, env_id: int = None):
        self.exec_remote('reset', proc=env_id)

    def is_done(self):
        return self.exec_remote('is_done')

    def collect(self, *args, **kwargs):
        return self.exec_remote('collect', args=args, kwargs=kwargs)

    def get_stats(self):
        return self.exec_remote('get_stats')
