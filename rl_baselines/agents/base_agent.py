import abc


class BaseAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def act(self):
        raise NotImplementedError

    @abc.abstractmethod
    def grad(self, _criterion):
        raise NotImplementedError
