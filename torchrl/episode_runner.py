import time
from torchrl.learners import BaseLearner
from torchrl.multi_proc_wrapper import MultiProcWrapper


DEFAULT_MAX_STEPS = int(1e6)


class EpisodeRunner:
    """
    EpisodeRunner is a utility wrapper to run episodes on a single Gym-like environment
    object. Each call to the `run()` method will run one episode for specified
    number of steps. Call `reset()` to reuse the same object again
    """
    def __init__(self, env, max_steps=DEFAULT_MAX_STEPS):
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

    def act(self, learner):
        """
        This routine is called from the `run` routine every time an action
        is needed for the environment to step. This function can be overridden
        in case the learner returns more than just the actions
        :param learner: An agent of type BaseLearner
        :return: Return the action(s) needed by the environment to act
        """
        action = learner.act(self._obs)
        return action

    def run(self, learner, steps=None, render=False, fps=30, store=False):
        """

        :param learner: An agent of type BaseLearner
        :param steps: Number of maximum steps in the current rollout
        :param render: Flag to render the environment, True or False
        :param fps: Rendering rate of the environment, frames per second if render is True
        :param store: Flag to store the history of the run
        :return: batch of transitions
        """
        assert not self._done, 'EpisodeRunner has ended. Call .reset() to reuse.'

        assert isinstance(learner, BaseLearner),\
            '"learner" should inherit from "BaseLearner", found invalid type "{}"'.format(type(learner))

        steps = steps or self.max_steps

        if render:
            self.env.render()
            time.sleep(1. / fps)

        obs_history = []
        action_history = []
        reward_history = []
        next_obs_history = []
        done_history = []

        while not self._done and steps:
            action = self.act(learner)
            next_obs, reward, self._done, info = self.env.step(action)

            if store:
                obs_history.append(self._obs)
                action_history.append(action)
                reward_history.append(reward)
                next_obs_history.append(next_obs)
                done_history.append(int(self._done))

            self._obs = next_obs
            steps -= 1

            if render:
                self.env.render()
                time.sleep(1. / fps)

        if store:
            return obs_history, action_history, reward_history, next_obs_history, done_history

    def stop(self):
        self.env.close()


class MultiEpisodeRunner(MultiProcWrapper):
    """
    This class is the parallel version of EpisodeRunner
    """
    def reset(self, env_id=None):
        self.exec_remote('reset', proc=env_id)

    def is_done(self):
        return self.exec_remote('is_done')

    def run(self, *args, **kwargs):
        return self.exec_remote('run', args=args, kwargs=kwargs)
