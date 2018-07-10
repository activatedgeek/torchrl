.. toctree::

.. _getting_started:

Getting Started
================

.. seealso::

    It might be helpful to take a look at :doc:`concepts`.

DQN with TorchRL
-----------------

In this tutorial, we will will utilize ``torchrl`` modules to
build a DQN experiment on the ``Gym`` environment ``CartPole-v1``

Register Problem
^^^^^^^^^^^^^^^^^

Each problem has an environment and a learning agent. We will utilize
the pre-built :class:`~torchrl.problems.dqn.DQNProblem` class by just
overriding these abstract methods. The full code for problem definition
is contained below.

.. code-block:: python
    :linenos:

    @registry.register_problem('dqn-cartpole-v1')
    class CartPoleDQNProblem(DQNProblem):
      def make_env(self):
        return gym.make('CartPole-v1')

      def init_agent(self):
        observation_space, action_space = self.get_gym_spaces()

        agent = CartPoleDQNAgent(
            observation_space,
            action_space,
            double_dqn=self.hparams.double_dqn,
            lr=self.hparams.actor_lr,
            gamma=self.hparams.gamma,
            target_update_interval=self.hparams.target_update_interval)

        return agent

Lastly, each such problem must be registered with a unique name
using the :meth:`~torchrl.registry.registry.register_problem` decorator as

.. code-block:: python

    @registry.register_problem('dqn-cartpole-v1')

It is then possible to use the CLI_ argument ``--problem=dqn-cartpole-v1``.

Register Hyperparameter Set
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before we discuss the problem above, we first define a new
:class:`~torchrl.registry.problems.HParams` object which is a
specification for the hyper-parameters of the problem. These
are arbitrary key-value pairs containing primitive values.

.. code-block:: python
    :linenos:

    @registry.register_hparam('dqn-cartpole')
    def hparam_dqn_cartpole():
      params = base_hparams.base_dqn()

      params.rollout_steps = 1
      params.num_processes = 1
      params.actor_lr = 1e-3
      params.gamma = 0.8
      params.target_update_interval = 5
      params.eps_min = 0.15
      params.buffer_size = 5000
      params.batch_size = 64
      params.num_total_steps = 10000

      return params

In this case, we start from a base hyperparameter set :meth:`~torchrl.problems.base_hparams.base_dqn`
provided by ``torchrl`` and override a few parameters like the learning rate
``params.actor_lr`` and discount factor ``params.gamma``. These are arbitrarily
defined and there is no restriction to the names as long as they are consistently
used.

Again, each such hyperparameter set must be registered
using a unique name using the :meth:`~torchrl.registry.registry.register_hparam`
decorator as

.. code-block:: python

    @registry.register_hparam('dqn-cartpole')

It is then possible to use the CLI_ argument ``--hparam-set=dqn-cartpole``.
This registry based approach makes hyperparameters composable and trackable
for reproducibility.

Create Environment
^^^^^^^^^^^^^^^^^^^

The :meth:`~torchrl.registry.problems.Problem.make_env` method provides
the specification on how to create an environment. In this case, we simply
create a new ``gym.Env`` object by passing the environment ID ``CartPole-v1``.

Initialize Agent
^^^^^^^^^^^^^^^^^

The :meth:`~torchrl.registry.problems.Problem.init_agent` method provides
the specification on how to create a new learning agent. This must return
a :class:`~torchrl.agents.base_agent.BaseAgent` object. The full code
is below. We base it off :class:`~torchrl.agents.dqn_agent.BaseDQNAgent`
from the codebase.

.. code-block:: python
    :linenos:

    class CartPoleDQNAgent(BaseDQNAgent):
      def compute_q_values(self, obs, action, reward, next_obs, done):
        for i, _ in enumerate(reward):
          if done[i] == 1:
            reward[i] = -1.0

        return super(CartPoleDQNAgent, self).compute_q_values(
            obs, action, reward, next_obs, done)

The agent created by :meth:`~torchrl.registry.problems.Problem.init_agent`
utilitizes an class instance attribute ``self.hparams`` which contains
the hyperparameter set object we created above.

This code also demonstrates how we can accomplish reward shaping. Note
how we have shaped the reward to be `-1` at the terminal state instead of `0`.


Run Experiment
^^^^^^^^^^^^^^^

We will use the ``torchrl`` CLI_ to run the experiment.

.. code-block:: bash

    torchrl --problem=dqn-cartpole-v1 \
            --hparam-set=dqn-cartpole \
            --seed=1 \
            --usr-dirs=experiments \
            --log-dir=log/dqn \
            --show-progress

Internally, this runs the :class:`~torchrl.episode_runner.MultiEpisodeRunner`
class.

``--problem`` and ``hparam-set`` arguments have been discussed before. A
summary of other arguments is below.

- ``--seed`` argument ensures reproducibility by calling the
  :meth:`~torchrl.utils.set_seeds` method.
- ``--usr-dirs`` argument ensures that the problems registered above are discoverable.
  This should be a comma-separated list of module folders.
- ``--log-dir`` is the directory that contains a dump of all hyperparameters
  from the hyperparameter set (including the base ones) and all the arguments
  for reproducibility like ``--seed``. It contains saved checkpoints so that
  experiments can be resumed later. It also contains the Tensorboard events file.
- ``--show-progress`` is a utility flag which shows current progress and estimated
  time remaining to completion.

.. warning::

    While reusing ``--log-dir``, make sure that the old events files are deleted
    to prevent any discrepancy in the Tensorboard dashboard.

The full list of options is available `below <CLI_>`_.

.. _CLI:

CLI Usage
----------

.. program-output:: torchrl -h


What's next?
-------------

See some ready-to-run :ref:`experiments`.
