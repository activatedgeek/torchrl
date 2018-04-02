import abc
import time


class Runner(metaclass=abc.ABCMeta):
    def __init__(self, env, agent, criterion, optimizer):
        self.env = env
        self.agent = agent
        self.criterion = criterion
        self.optimizer = optimizer

    @abc.abstractmethod
    def step(self, state):
        raise NotImplementedError

    def run_episode(self, **kwargs):
        max_steps = kwargs.get('max_steps', None)
        render = kwargs.get('render', False)
        fps = kwargs.get('fps', 30)

        history = []
        steps = 0

        state = self.env.reset()
        if render:
            self.env.render()
            time.sleep(1. / fps)

        while True:
            transition = self.step(state)
            history.append(transition)

            steps += 1

            if render:
                self.env.render()
                time.sleep(1. / fps)

            if transition.done or (max_steps and steps >= max_steps):
                break

            state = transition.next_state

        return history

    def run(self, num_episodes, **kwargs):
        store_history = kwargs.get('store_history', False)

        history = [] if store_history else None
        for _ in range(num_episodes):
            episode = self.run_episode(**kwargs)
            if store_history and episode:
                history.append(episode)

        return history
