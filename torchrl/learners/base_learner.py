import abc
from ..agents import BaseAgent


class BaseLearner(metaclass=abc.ABCMeta):
    """
    This is base runner specification which can encapsulate everything
    how a Reinforcement Learning Algorithm would function.
    """
    def __init__(self, agent, criterion, optimizer):
        assert isinstance(agent, BaseAgent),\
            '"agent" should inherit from "BaseAgent", found invalid type "{}"'.format(type(agent))

        self.agent = agent
        self.criterion = criterion
        self.optimizer = optimizer

    @abc.abstractmethod
    def act(self, *args, **kwargs):
        """
        This is the method that should be called at every step of the episode
        :param state: Representation of the state
        :return: identity of the action to be taken, as desired by the environment
        """
        raise NotImplementedError

    @abc.abstractmethod
    def transition(self, episode_id, state, action, reward, next_state, done):
        """
        This routine can be used to handle the transition information returned by the
        environment and is provided as a utility routine and will be called by
        the `EpisodeRunner` after every step

        :param episode_id:
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :param done:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        """
        This is the method to be called when the internal parameters need to be updated
        """
