import abc


class BaseAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def act(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def grad(self, _criterion):
        raise NotImplementedError
