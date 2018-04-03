from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, size=2000):
        self.memory = deque(maxlen=size)

    def extend(self, transitions):
        self.memory.extend(transitions)

    def add(self, transition):
        assert isinstance(transition, Transition), \
            'Input should be an instance of Transition, found {}'.format(str(type(transition)))
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
