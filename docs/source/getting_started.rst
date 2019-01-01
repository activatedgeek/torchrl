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

    @registry.register_problem
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

    @registry.register_problem
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

Again, each such hyperparameter set must be registered
using a unique name using the :meth:`~torchrl.registry.registry.register_hparam`
decorator as

.. code-block:: python

    @registry.register_hparam # or @registry.register_hparam('my_hparam_set')

For ease of use, ``torchrl`` automatically registers and ``static``
methods of a Problem class which start with ``hparams_``. This also adds an
extra association with the problem which is helpful to discover all hyper-parameter
sets associated with a problem. The HParams set is registered without the
``hparams_`` prefix.

It is then possible to use the CLI argument ``--hparam-set=dqn_cartpole``
or ``--hparam-set=double_dqn_cartpole``. This registry based approach makes
hyperparameters composable and trackable for reproducibility.

Register Problem
^^^^^^^^^^^^^^^^^

Lastly, each such problem must be registered using the
:meth:`~torchrl.registry.registry.register_problem` decorator as

.. code-block:: python

    @registry.register_problem

This will take the class name, convert it to camel case and store
in the registry. One can also optionally provide the name as

.. code-block:: python

    @registry.register_problem('my_dqn_problem')

It is then possible to use the CLI argument ``--problem=dqn_cartpole``
(or ``--problem=my_dqn_problem`` if custom name used).


Full Code
^^^^^^^^^^

The full code to run the experiment is as simple as below - less than 50 lines.

.. code-block:: python
    :linenos:

    from torchrl import registry
    from torchrl import utils
    from torchrl.problems import base_hparams, DQNProblem
    from torchrl.agents import BaseDQNAgent


    @registry.register_problem
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

We will use the CLI to run the experiment.

.. code-block:: bash

    torchrl --problem=dqn_cartpole \
            --hparam-set=dqn_cartpole \
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
  This is optional and if unspecified, no files are written.
- ``--show-progress`` is a utility flag which shows current progress and estimated
  time remaining to completion.

.. warning::

    While reusing ``--log-dir``, make sure that the old events files are deleted
    to prevent any discrepancy in the ``Tensorboard`` dashboard.

The full list of options is available at :doc:`cli`.
