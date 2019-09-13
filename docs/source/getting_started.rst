.. _getting_started:

Getting Started
================

.. seealso::

    It might be helpful to take a look at :doc:`concepts`.

DQN with TorchRL
-----------------

In this tutorial, we will will utilize ``torchrl`` modules to
build a DQN experiment on the ``Gym`` environment ``CartPole-v1``

Initialize Agent
^^^^^^^^^^^^^^^^^

Each problem has an environment and a learning agent. We utilize
the pre-built :class:`~torchrl.problems.dqn.DQNProblem` class by just
overriding some abstract methods. A rough sketch of the code to
initialize the agent is below. Note that the contents of this method
are completely upto the user as long as it returns a valid
:class:`~torchrl.agents.base_agent.BaseAgent`.

.. code-block:: python
    :linenos:

    class DQNCartpole(DQNProblem):
      def init_agent(self):
        observation_space, action_space = utils.get_gym_spaces(self.runner.make_env)

        agent = BaseDQNAgent(
            observation_space,
            action_space,
            double_dqn=self.hparams.double_dqn,
            lr=self.hparams.actor_lr,
            gamma=self.hparams.gamma,
            target_update_interval=self.hparams.target_update_interval)

        return agent

      ...

Register Hyperparameter Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We then define a new :class:`~torchrl.registry.problems.HParams` object
which is a specification for the hyper-parameters of the problem. These
are arbitrary key-value pairs containing primitive values.

.. code-block:: python
    :linenos:

    class DQNCartpole(DQNProblem):

      ...

      @staticmethod
      def hparams_dqn_cartpole():
        params = base_hparams.base_dqn()

        params.env_id = 'CartPole-v1'

        params.rollout_steps = 1
        params.num_processes = 1
        params.actor_lr = 1e-3
        params.gamma = 0.99
        params.target_update_interval = 10
        params.eps_min = 1e-2
        params.buffer_size = 1000
        params.batch_size = 32
        params.num_total_steps = 10000
        params.num_eps_steps = 500

        return params

      @staticmethod
      def hparams_double_dqn_cartpole():
        params = DQNCartpole.hparams_dqn_cartpole()

        params.double_dqn = True
        params.target_update_interval = 5

        return params

In this case, we start from a base hyperparameter set :meth:`~torchrl.problems.base_hparams.base_dqn`
provided by ``torchrl`` and override a few parameters like the learning rate
``params.actor_lr`` and discount factor ``params.gamma``. These are arbitrarily
defined and there is no restriction to the names as long as they are consistently
used.

Full Code
^^^^^^^^^^

The full code to run the experiment is as simple as below - less than 50 lines.

.. code-block:: python
    :linenos:

    from torchrl import utils
    from torchrl.problems import base_hparams, DQNProblem
    from torchrl.contrib.problems import DQNProblem
    from torchrl.contrib.agents import BaseDQNAgent


    class DQNCartpole(DQNProblem):
      def init_agent(self):
        observation_space, action_space = utils.get_gym_spaces(self.runner.make_env)

        agent = BaseDQNAgent(
            observation_space,
            action_space,
            double_dqn=self.hparams.double_dqn,
            lr=self.hparams.actor_lr,
            gamma=self.hparams.gamma,
            target_update_interval=self.hparams.target_update_interval)

        return agent

      @staticmethod
      def hparams_dqn_cartpole():
        params = base_hparams.base_dqn()

        params.env_id = 'CartPole-v1'

        params.rollout_steps = 1
        params.num_processes = 1
        params.actor_lr = 1e-3
        params.gamma = 0.99
        params.target_update_interval = 10
        params.eps_min = 1e-2
        params.buffer_size = 1000
        params.batch_size = 32
        params.num_total_steps = 10000
        params.num_eps_steps = 500

        return params

      @staticmethod
      def hparams_double_dqn_cartpole():
        params = DQNCartpole.hparams_dqn_cartpole()

        params.double_dqn = True
        params.target_update_interval = 5

        return params


Run Experiment
^^^^^^^^^^^^^^^


```
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args=dict(
    seed=1,
    log_interval=1000,
    eval_interval=1000,
    num_eval=1,
)

dqn_cartpole = DQNCartpole(
    hparams_dqn_cartpole(),
    argparse.Namespace(**args),
    None, # Disable logging
    device=device,
    show_progress=True,
)

dqn_cartpole.run()
```