from typing import Callable, Union
from ..utils.misc import to_camel_case
from .problems import Problem

__all__ = [
    'register_hparam',
    'list_hparams',
    'get_hparam',
    'remove_hparam',
    'register_problem',
    'list_problems',
    'get_problem',
    'remove_problem',
    'list_problem_hparams',
    'get_problem_hparam',
]


_ALL_HPARAMS = {}
_ALL_PROBLEMS = {}
_PROBLEM_HPARAMS = {}

def _common_decorator(func: Callable, registration_name: str,
                      target_dict: dict):
  registration_name = to_camel_case(registration_name)

  assert registration_name not in target_dict, \
    'Attempt to re-register "{}"'.format(registration_name)

  target_dict[registration_name] = func

  if issubclass(func, Problem):
    ##
    # Auto-register any class static methods starting with "hparams_"
    # as hyper-parameter sets
    #
    class_hparams = [
        (to_camel_case(attr[8:]), getattr(func, attr)) for attr in dir(func)
        if attr.startswith('hparams_') and attr in func.__dict__ and
        isinstance(func.__dict__[attr], staticmethod)
    ]

    class_hparam_ids = []
    for hparam_set_id, target_method in class_hparams:
      _ALL_HPARAMS[hparam_set_id] = target_method
      class_hparam_ids.append(hparam_set_id)
    _PROBLEM_HPARAMS[registration_name] = class_hparam_ids

  return func


def _common_list(target_dict: dict):
  return list(target_dict.keys())


def _common_get(target_dict: dict, key: str):
  return target_dict[key]


def _common_remove(target_dict: dict, key: str):
  target_dict.pop(key)


def register_hparam(name: Union[Callable, str]):
  """
  A decorator to register hyperparameter function.

  Example:

      .. code-block:: python

          import torch.registry as registry

          @registry.register_hparam
          def my_new_hparams():
            hparams = registry.HParams()
            hparams.x = 1
            return hparams

      This will be registered by name `my_new_hparams`.
      Optionally, we can also provide a name as argument
      to the decorator.

      .. code-block:: python

          @registry.register_hparam('my_renamed_hparams')
  Args:
      name (str, :class:`~typing.Callable`): Optionally pass a string
          argument for name or will be the callable.

  Returns:
      :class:`~typing.Callable`: A decorated function.
  """
  target_dict = _ALL_HPARAMS

  if callable(name):
    return _common_decorator(name, name.__name__, target_dict)

  return lambda func: _common_decorator(func, name, target_dict)


def register_problem(name: Union[Callable, str]):
  """
  A decorator to register problems.

  Example:

      .. code-block:: python

          import torch.registry as registry

          @registry.register_problem
          class MyProblem(registry.Problem):
            ...

      This will be registered by name `my_problem`.
      Optionally, we can also provide a name as argument
      to the decorator.

      .. code-block:: python

          @registry.register_problem('my_renamed_problem')
  Args:
      name (str, :class:`~typing.Callable`): Optionally pass a string
          argument for name or will be the callable.

  Returns:
      :class:`~typing.Callable`: A decorated function.
  """
  target_dict = _ALL_PROBLEMS

  if callable(name):
    return _common_decorator(name, name.__name__, target_dict)

  return lambda func: _common_decorator(func, name, target_dict)


def list_hparams() -> list:
  """
  List all registered hyperparameters.

  Returns:
      list: List of hyperparameter name strings.
  """
  return _common_list(_ALL_HPARAMS)


def get_hparam(hparam_set_id: str) -> Callable:
  """
  Get registered hyperparameter by name.

  Args:
      hparam_set_id (str): A string representing name of hyperparameter set.

  Returns:
      :class:`~typing.Callable`: A function that returns \
        :class:`~torchrl.registry.problems.HParams`.
  """
  return _common_get(_ALL_HPARAMS, hparam_set_id)


def list_problem_hparams():
  """
  List all registered hyperparameters associated with a problem.
  Any static method of a Problem class whose name is prefixed
  with `hparams_` is associated to a problem. This routine
  returns all such associations available.

  Example:
      The format of returned values is

      .. code-block:: json

          {
            "problem_name": [
              "hparam_set1", "hparam_set2"
            ],
            "other_problem": [
              "other_problem_hparam1"
            ]
          }

  Returns:
      list: List of problem-hyperparameter associations of the following format.
  """
  return _PROBLEM_HPARAMS


def get_problem_hparam(problem_id: str):
  """
  Get the associated hyperparameters to a problem.

  Args:
      problem_id (str): Name of registered problem.

  Returns:
      list: List of hyperparameter sets.
  """
  return _PROBLEM_HPARAMS[problem_id]


def remove_hparam(hparam_set_id: str):
  """
  De-register a hyperparameter set.

  Args:
      hparam_set_id (str): Name of registered hyperparameter.
  """
  _common_remove(_ALL_HPARAMS, hparam_set_id)


def list_problems():
  """
  List all registered Problems.

  Returns:
      list: List of string containing all problem names.
  """
  return _common_list(_ALL_PROBLEMS)


def get_problem(problem_id: str):
  """
  Get uninstatiated problem class.

  Args:
      problem_id (str): Name of registered problem.

  Returns:
      :class:`torchrl.registry.problems.Problem`: Any derived problem class.
  """
  return _common_get(_ALL_PROBLEMS, problem_id)


def remove_problem(problem_id: str):
  """
  De-register a problem.

  Args:
      problem_id (str): Name of registered problem.
  """
  _common_remove(_ALL_PROBLEMS, problem_id)
