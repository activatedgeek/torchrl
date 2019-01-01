Core Concepts
==============

This document is a glossary for core concepts of *TorchRL* framework.

.. _Agent:

Agent
--------------

:class:`~torchrl.agents.base_agent.BaseAgent` is an abstract
class which defines the learning agent in the given Environment_.

.. _Environment:

Environment
-------------

Environment is the system which provides feedback to the Agent_. Currently,
Open AI `gym.Env` environments are being used. The system is flexible enough
to extend to any other environment kind.


.. _Problem:

Problem
--------

Any task is defined by extending the abstract class
:class:`~torchrl.registry.problems.Problem`. A problem's entrypoint
is :meth:`~torchrl.registry.problems.Problem.run` which generates
the trajectory rollout and call's the Agent_'s
:meth:`~torchrl.agents.base_agent.BaseAgent.learn` method with
appropriate rollout information.


Hyper-Parameter Set
--------------------

A :class:`~torchrl.registry.problems.HParams` set is a class of arbitrary
key-value pairs that contain the hyper-parameters for the problem. Keeping
these as first-class objects in the code base allow for easily reproducible
experiments.

Runner
-------

A :class:`~torchrl.episode_runner.MultiEpisodeRunner` takes in a
method which returns a constructed environment and creates multiple
subprocess copies for parallel trajectory rollouts via the
:meth:`~torchrl.episode_runner.MultiEpisodeRunner.collect` method. Each
Problem_ internally creates a :class:`~torchrl.episode_runner.MultiEpisodeRunner`
and executes the collection process.
