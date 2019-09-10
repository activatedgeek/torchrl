.. _experiments:

Experiments
============

A few ready-to-run experiments using the command line interface.

.. warning::

    Make sure you have the ``experiments`` folder from the
    Github repository `activatedgeek/torchrl <//github.com/activatedgeek/torchrl>`_.


.. note::

    See :doc:`cli` for further extensions of these commands.

List all Problems
-----------------

This lists each problem and associated hyperparameter set.

.. code-block:: bash

    torchrl --usr-dirs experiments list -o yaml problems

.. program-output:: torchrl --usr-dirs ../../experiments list -o yaml problems


We use one of the problems from above as an example.

Run DQN on CartPole
--------------------

.. code-block:: bash

    torchrl --usr-dirs experiments run dqn_cartpole \
            --hparam-set dqn_cartpole \
            --seed 1 \
            --log-dir log/dqn \
            --progress

Resume Experiment
++++++++++++++++++

To resume this experiment, we simply point to the log directory,

.. code-block:: bash

    torchrl --usr-dirs experiments resume log/dqn --progress


Extra Hyperparameters
++++++++++++++++++++++

Extra hyperparameters can be provided as arbitrary key value pairs
multiple times and can be accessed inside the Problem

.. code-block:: bash

    torchrl --usr-dirs experiments run dqn_cartpole \
            --hparam-set dqn_cartpole \
            --extra-hparams "num_total_steps=3000" \
            --extra-hparams "lr=0.0001" \
            --seed 1 \
            --log-dir log/dqn \
            --progress
