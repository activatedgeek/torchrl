import abc


class BaseLearner(metaclass=abc.ABCMeta):
    """
    This is base runner specification which can encapsulate everything
    how a Reinforcement Learning Algorithm would function.
    """
    def __init__(self, optimizer):
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
    def transition(self, state, action, reward, next_state, done, episode_id=None):
        """
        This routine can be used to handle the transition information returned by the
        environment. All inputs should be batches of information.

        :param state: list of states
        :param action: list of actions
        :param reward: list of rewards
        :param next_state: list of next_states
        :param done: list of done
        :param episode_id: list of episode_id(s)
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        """
        This method represents the learning step
        """
        raise NotImplementedError
