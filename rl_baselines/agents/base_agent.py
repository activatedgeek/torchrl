import abc


class BaseAgent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def act(self, state):
        raise NotImplementedError
