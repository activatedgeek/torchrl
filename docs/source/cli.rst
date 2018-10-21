CLI Reference
=============

This document is a reference of all the CLI commands available.
Note all the CLI arguments must follow strict hierarchy.

.. note::

    The CLI has been built using `click <http://click.pocoo.org/5/>`_.

.. program-output:: torchrl --help

list
----

This sub-command is used to list registered :class:`~torchrl.registry.problems.Problem`s
and :class:`~torchrl.registry.problems.HParams` sets.

.. program-output:: torchrl list --help

.. warning::

    Due to strict options ordering, ``--usr-dirs`` belong to the
    top level command. Hence, the correct usage is

    .. code:: shell

        torchrl --usr-dirs experiments list problems

run
---

This sub-command is used to run the actual experiments.

.. program-output:: torchrl run --help


resume
------

This sub-command is used to resume experiments from an
experiment log directory.

.. program-output:: torchrl resume --help
