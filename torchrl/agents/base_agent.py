import abc


class BaseAgent(metaclass=abc.ABCMeta):
    """
    Abstract class to represent the agent
    """

    @abc.abstractmethod
    def forward(self, state):
        """
        Returns the raw vector result (which may be used by the `act` method below
        to decide what action to take)

        :param state:
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, state):
        """
        This method must be called to ask the agent to return the action to be
        take for a given state

        :param state: Representation of the state
        :return: identity of the action to be taken, as desired by the environment
        """
        raise NotImplementedError
