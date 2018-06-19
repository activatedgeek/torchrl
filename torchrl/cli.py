import argparse
import os
import sys
import torch
import importlib

import torchrl.registry as registry


def parse_args(argv):
  parser = argparse.ArgumentParser(prog='RL Experiment Runner',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument('--problem', type=str, required=True,
                      help='Problem name')
  parser.add_argument('--hparam-set', type=str, required=True,
                      help='Hyperparameter set name')
  parser.add_argument('--seed', type=int, metavar='', help='Random seed')

  parser.add_argument('--usr-dirs', type=str, metavar='', default='',
                      help='Comma-separated list of user module directories')

  parser.add_argument('--cuda', dest='cuda', action='store_true',
                      help='Enable CUDA')
  parser.add_argument('--no-cuda', dest='cuda', action='store_false',
                      help='Disable CUDA')
  parser.set_defaults(cuda=True)

  parser.add_argument('--log-dir', type=str, metavar='', default='log',
                      help='Directory to store logs')
  parser.add_argument('--save-dir', type=str, metavar='',
                      help='Directory to store agent')
  parser.add_argument('--load-dir', type=str, metavar='',
                      help='Directory to load agent')

  parser.add_argument('--log-interval', type=int, metavar='', default=100,
                      help='Log interval w.r.t epochs')
  parser.add_argument('--eval-interval', type=int, metavar='', default=500,
                      help='Eval interval w.r.t epochs')
  parser.add_argument('--num-eval', type=int, metavar='', default=10,
                      help='Number of evaluations')

  args = parser.parse_args(args=argv)

  if not torch.cuda.is_available():
    args.cuda = False

  return args


def import_usr_dir(usr_dir):
  dir_path = os.path.abspath(os.path.expanduser(usr_dir).rstrip("/"))
  containing_dir, module_name = os.path.split(dir_path)
  sys.path.insert(0, containing_dir)
  importlib.import_module(module_name)
  sys.path.pop(0)


def main():
  args = parse_args(sys.argv[1:])

  # Import external modules containing problems
  for usr_dir in args.usr_dirs.split(','):
    import_usr_dir(usr_dir)

  hparams = registry.get_hparam(args.hparam_set)()
  problem_cls = registry.get_problem(args.problem)
  problem = problem_cls(hparams, args)
  problem.run()


if __name__ == '__main__':
  main()
