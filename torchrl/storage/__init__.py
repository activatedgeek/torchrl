from collections import namedtuple
from .replay_buffer import ReplayBuffer
from .prioritized_replay_buffer import PrioritizedReplayBuffer


Transition = namedtuple('Transition', [
    'obs',
    'action',
    'reward',
    'next_obs',
    'done',
])
