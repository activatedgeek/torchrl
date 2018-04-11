from collections import deque, namedtuple
import random

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


class Episode:
    def __init__(self):
        self._history = []

    def append(self, state, action, reward, next_state, done):
        self._history.append(Transition(state, action, reward, next_state, done))

    def __iter__(self):
        for transition in self._history:
            yield transition

    def __len__(self):
        return len(self._history)

    def __getitem__(self, index):
        return self._history[index]


class EpisodeBuffer:
    def __init__(self, memory=None):
        self.memory = memory

        self._episodes = []

    def append(self, episode):
        self._episodes.append(episode)

    def extend(self, *episodes):
        self._episodes.extend(episodes)

    def clear(self):
        self._episodes.clear()

    def __len__(self):
        return len(self._episodes)


class ReplayMemory:
    def __init__(self, size=100000):
        self.memory = deque(maxlen=size)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __iter__(self):
        for transition in self.memory:
            yield transition

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]
