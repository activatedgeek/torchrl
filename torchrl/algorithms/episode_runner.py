import abc
import time


class EpisodeRunner(metaclass=abc.ABCMeta):
    def __init__(self, env, agent, criterion, optimizer):
        self.env = env
        self.agent = agent
        self.criterion = criterion
        self.optimizer = optimizer

    @abc.abstractmethod
    def step(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def process_episode(self, episode):
        raise NotImplementedError

    @abc.abstractmethod
    def learn(self):
        raise NotImplementedError

    def run_episode(self, **kwargs):
        max_steps = kwargs.get('max_steps', 1000000)
        render = kwargs.get('render', False)
        fps = kwargs.get('fps', 30)

        episode = []
        steps = 0

        state = self.env.reset()
        if render:
            self.env.render()
            time.sleep(1. / fps)

        for step in range(1, max_steps + 1):
            transition = self.step(state)
            episode.append(transition)

            steps += 1

            if render:
                self.env.render()
                time.sleep(1. / fps)

            if transition.done:
                break

            state = transition.next_state

        return episode

    def run(self, num_episodes, **kwargs):
        store_history = kwargs.get('store_history', False)

        history = [] if store_history else None
        for eid in range(1, num_episodes + 1):
            episode = self.run_episode(**kwargs)
            self.process_episode(episode)
            if store_history:
                history.append(episode)

        return history
