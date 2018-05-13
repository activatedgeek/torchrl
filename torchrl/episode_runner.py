import abc
import time
import gym
import numpy as np
from torchrl.learners import BaseLearner
from torchrl.multi_proc_wrapper import MultiProcWrapper


DEFAULT_MAX_STEPS = int(1e6)


class EpisodeRunner(metaclass=abc.ABCMeta):
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

        self.reset()

    def reset(self):
        """
        Reset internal state for the `run` method to be reused
        """
        self._obs = self.env.reset()
        self._done = False

    def is_done(self):
        """
        :return: True if the episode has ended, False otherwise
        """
        return self._done

    @abc.abstractmethod
    def act(self, learner: BaseLearner):
        """
        This routine is called from the `run` routine every time an action
        is needed for the environment to step.
        """
        raise NotImplementedError

    def run(self, learner: BaseLearner, steps: int = None, render: bool = False, fps: int = 30, store: bool = False):
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

        obs_history, action_history, reward_history, next_obs_history, done_history = \
            self.init_run_history(self.env.observation_space, self.env.action_space)

        while not self._done and steps:
            action = self.act(learner)
            next_obs, reward, self._done, _ = self.env.step(action)

            if store:
                obs_history = np.append(obs_history, np.expand_dims(self._obs, axis=0), axis=0)
                action_history = np.append(action_history, np.array([[action]]), axis=0)
                reward_history = np.append(reward_history, np.array([[reward]]), axis=0)
                next_obs_history = np.append(next_obs_history, np.expand_dims(next_obs, axis=0), axis=0)
                done_history = np.append(done_history, np.array([[int(self._done)]]), axis=0)

            self._obs = next_obs
            steps -= 1

            if render:
                self.env.render()
                time.sleep(1. / fps)

        if store:
            return obs_history, action_history, reward_history, next_obs_history, done_history

    def stop(self):
        self.env.close()

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


class MultiEpisodeRunner(MultiProcWrapper):
    """
    This class is the parallel version of EpisodeRunner
    """
    def reset(self, env_id: int = None):
        self.exec_remote('reset', proc=env_id)

    def is_done(self):
        return self.exec_remote('is_done')

    def run(self, *args, **kwargs):
        return self.exec_remote('run', args=args, kwargs=kwargs)
