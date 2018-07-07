.. _experiments:

Experiments
============

A few ready-to-run experiments using the command line interface.

.. warning::

    Make sure you have the ``experiments`` folder from the
    Github repository `activatedgeek/torchrl <//github.com/activatedgeek/torchrl>`_.

DQN on CartPole-v1
-------------------

.. code-block:: bash

    torchrl --problem=dqn-cartpole-v1 \
            --hparam-set=dqn-cartpole \
            --seed=1 \
            --usr-dirs=experiments \
            --log-dir=log/dqn \
            --show-progress

Double DQN on CartPole-v1
--------------------------

.. code-block:: bash

    torchrl --problem=dqn-cartpole-v1 \
            --hparam-set=ddqn-cartpole \
            --seed=1 \
            --usr-dirs=experiments \
            --log-dir=log/ddqn \
            --show-progress

Prioritized DQN on CartPole-v1
-------------------------------

.. code-block:: bash

    torchrl --problem=prioritized-dqn-cartpole-v1 \
            --hparam-set=dqn-cartpole \
            --seed=1 \
            --usr-dirs=experiments \
            --log-dir=log/dqn \
            --show-progress

Prioritized Double DQN on CartPole-v1
--------------------------------------

.. code-block:: bash

    torchrl --problem=prioritized-dqn-cartpole-v1 \
            --hparam-set=ddqn-cartpole \
            --seed=1 \
            --usr-dirs=experiments \
            --log-dir=log/dqn \
            --show-progress

A2C on CartPole-v0
-------------------

.. code-block:: bash

    torchrl --problem=a2c-cartpole-v0 \
            --hparam-set=a2c-cartpole \
            --seed=1 \
            --usr-dirs=experiments \
            --log-dir=log/a2c \
            --show-progress


DDPG on Pendulum-v0
--------------------

.. code-block:: bash

    torchrl --problem=ddpg-pendulum-v0 \
            --hparam-set=ddpg-pendulum \
            --seed=1 \
            --usr-dirs=experiments \
            --log-dir=log/ddpg \
            --show-progress


PPO on Pendulum-v0
-------------------

.. code-block:: bash

    torchrl --problem=ppo-pendulum-v0 \
            --hparam-set=ppo-pendulum \
            --seed=1 \
            --usr-dirs=experiments \
            --log-dir=log/ppo \
            --show-progress


Resume Experiments
===================

To reload an experiment from previous run, say for instance the DQN run
above for `3000` more steps (optional argumen),

.. code-block:: bash

    torchrl --load-dir=log/dqn \
            --extra-hparams="num_total_steps=3000" \
            --usr-dirs=experiments \
            --show-progress

This will read all the other parameters from the directory and load the latest
checkpoint.
