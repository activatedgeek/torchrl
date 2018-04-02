import time
import random
from ..memory import Transition


class Runner:
    def __init__(self, env, agent, criterion, optimizer):
        self.env = env
        self.agent = agent
        self.criterion = criterion
        self.optimizer = optimizer

    def step(self, state):
        action = random.randrange(self.env.action_space.n)
        next_state, reward, done, info = self.env.step(action)
        transition = Transition(state, action, reward, next_state, done)
        return transition

    def run_episode(self, **kwargs):
        max_steps = kwargs.get('max_steps', 1000000)
        render = kwargs.get('render', False)
        fps = kwargs.get('fps', 30)

        history = []
        steps = 0

        state = self.env.reset()
        if render:
            self.env.render()
            time.sleep(1. / fps)

        for step in range(1, max_steps + 1):
            transition = self.step(state)
            history.append(transition)

            steps += 1

            if render:
                self.env.render()
                time.sleep(1. / fps)

            if transition.done:
                break

            state = transition.next_state

        return history

    def run(self, num_episodes, **kwargs):
        store_history = kwargs.get('store_history', False)

        history = [] if store_history else None
        for _ in range(num_episodes):
            episode = self.run_episode(**kwargs)
            if store_history:
                history.append(episode)

        return history
