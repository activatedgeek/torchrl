import argparse
import ast
import os
import sys
import torch
import importlib

import torchrl.registry as registry


def import_usr_dir(usr_dir):
  dir_path = os.path.abspath(os.path.expanduser(usr_dir).rstrip("/"))
  containing_dir, module_name = os.path.split(dir_path)
  sys.path.insert(0, containing_dir)
  importlib.import_module(module_name)
  sys.path.pop(0)


def parse_args(argv):
  parser = argparse.ArgumentParser(prog='RL Experiment Runner', formatter_class=argparse.ArgumentDefaultsHelpFormatter)  # pylint: disable=line-too-long

  parser.add_argument('--problem', type=str, metavar='', default='',
                      help='Problem name')
  parser.add_argument('--hparam-set', type=str, metavar='', default='',
                      help='Hyperparameter set name')
  parser.add_argument('--extra-hparams', type=str, metavar='', default='',
                      help="""Comma-separated list of extra key-value pairs,
                      automatically handles types int/float/str""")
  parser.add_argument('--seed', type=int, metavar='', help='Random seed')
  parser.add_argument('--progress', action='store_true', dest='progress',
                      help='Show epoch progress')
  parser.add_argument('--no-cuda', dest='no_cuda', action='store_true',
                      help='Disable CUDA')
  parser.add_argument('--device', type=str, metavar='', default='cuda',
                      help='Device selection for GPU')

  parser.add_argument('--usr-dirs', type=str, metavar='', default='',
                      help='Comma-separated list of user module directories')
  parser.add_argument('--log-dir', type=str, metavar='', default='log',
                      help='Directory to store logs')
  parser.add_argument('--load-dir', type=str, metavar='',
                      help='Directory to load agent')

  parser.add_argument('--log-interval', type=int, metavar='', default=100,
                      help='Log interval w.r.t epochs')
  parser.add_argument('--eval-interval', type=int, metavar='', default=1000,
                      help='Eval interval w.r.t epochs')
  parser.add_argument('--num-eval', type=int, metavar='', default=10,
                      help='Number of evaluations')

  args = parser.parse_args(args=argv)

  # Import external modules from user directories
  if args.usr_dirs:
    for usr_dir in args.usr_dirs.split(','):
      import_usr_dir(usr_dir)
  del args.usr_dirs

  # Setup CUDA devices
  args.cuda = not args.no_cuda and torch.cuda.is_available()
  if not args.cuda:
    args.device = 'cpu'
  del args.no_cuda

  # Parse extra hyper parameters with
  # auto type conversion to string/int/float
  if args.extra_hparams:
    def handle_extra_hparams(pair_str):
      key, value = pair_str.split('=', 1)
      try:
        value = ast.literal_eval(value)
      except ValueError:
        pass
      return key, value

    args.extra_hparams = dict(map(handle_extra_hparams,
                                  args.extra_hparams.split(',')))
  else:
    args.extra_hparams = {}

  # Expand to absolute paths
  if args.log_dir:
    args.log_dir = os.path.abspath(args.log_dir)
  if args.load_dir:
    args.load_dir = os.path.abspath(args.load_dir)

  return args


def main():
  args = parse_args(sys.argv[1:])

  if args.load_dir:
    params, loaded_args = registry.problems.Problem.load_from_dir(args.load_dir)
    args.__dict__.update(loaded_args.__dict__)
  else:
    params = registry.get_hparam(args.hparam_set)()
    params.update(args.extra_hparams)

  problem_cls = registry.get_problem(args.problem)
  problem = problem_cls(params, args)
  if args.load_dir:
    problem.load_latest_checkpoint(args.load_dir)
  problem.run()


if __name__ == '__main__':
  main()
