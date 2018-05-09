import abc


class BaseLearner(metaclass=abc.ABCMeta):
    """
    This is base runner specification which can encapsulate everything
    how a Reinforcement Learning Algorithm would function.
    """
    def __init__(self):
        self.is_cuda = False
        self.training = False

    @abc.abstractmethod
    def act(self, *args, **kwargs):
        """
        This is the method that should be called at every step of the episode
        :param state: Representation of the state
        :return: identity of the action to be taken, as desired by the environment
        """
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        """
        This method represents the learning step
        """
        raise NotImplementedError

    @abc.abstractmethod
    def cuda(self):
        """
        Enable CUDA on the Learner
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def train(self):
        """
        Enable train mode for the agent (helpful with NN agents with BatchNorm, Dropout etc.)
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def eval(self):
        """
        Enable evaluation mode for the agent (helpful with NN agents with BatchNorm, Dropout etc.)
        :return:
        """
        raise NotImplementedError

    def reset(self):
        """
        Optional function to reset learner's internals
        :return:
        """
        pass

    @abc.abstractmethod
    def save(self, dir):
        """
        Store the agent for future usage
        :param dir: The directory to save arbitrary files to
        :return:
        """
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, dir):
        """
        Load a pre-trained agent
        :param dir: The directory from a pre-trained agent (must be inverse of save)
        :return:
        """
        raise NotImplementedError
