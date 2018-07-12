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
  target_dict = _ALL_HPARAMS

  if callable(name):
    return _common_decorator(name, name.__name__, target_dict)

  return lambda func: _common_decorator(func, name, target_dict)


def register_problem(name: Union[Callable, str]):
  target_dict = _ALL_PROBLEMS

  if callable(name):
    return _common_decorator(name, name.__name__, target_dict)

  return lambda func: _common_decorator(func, name, target_dict)


def list_hparams():
  """List all Hyperparameter Sets in registry."""
  return _common_list(_ALL_HPARAMS)


def get_hparam(hparam_set_id: str):
  """Get arbitrary HParam set from registry."""
  return _common_get(_ALL_HPARAMS, hparam_set_id)


def list_problem_hparams():
  """List HParam sets, problem-compatible."""
  return _PROBLEM_HPARAMS


def get_problem_hparam(problem_id: str):
  """List HParam sets, problem-compatible."""
  return _PROBLEM_HPARAMS[problem_id]


def remove_hparam(hparam_set_id: str):
  """Remove arbitrary HParam set from registry."""
  return _common_remove(_ALL_HPARAMS, hparam_set_id)


def list_problems():
  """List all registered Problems."""
  return _common_list(_ALL_PROBLEMS)


def get_problem(problem_id: str):
  """Get arbitrary Problem from registry."""
  return _common_get(_ALL_PROBLEMS, problem_id)


def remove_problem(problem_id: str):
  """Remove arbitrary Problem from registry."""
  return _common_remove(_ALL_PROBLEMS, problem_id)
