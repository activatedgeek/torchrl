import argparse
import ast
import click
import os
from ruamel import yaml
import torch
import warnings


from ... import registry


class ExtraHParamType(click.ParamType):

  def convert(self, value, param, ctx):
    try:
      key, val = value.split('=', 1)
      try:
        val = ast.literal_eval(val)
      except ValueError:
        pass

      ctx.obj['extra_hparams'][key] = val

      return [key, val]
    except ValueError:
      self.fail('%s is not a input' % value, param, ctx)


def do_run(problem,
           hparam_set: str = None,
           extra_hparams: dict = None,
           progress: bool = False,
           cuda: bool = False,
           device: str = None,
           log_dir: str = None,
           resume: bool = False,
           start_epoch: int = None,
           **kwargs):
  problem_cls = registry.get_problem(problem)
  if not hparam_set:
    hparam_set_list = registry.get_problem_hparam(problem)
    assert hparam_set_list
    hparam_set = hparam_set_list[0]

  hparams = registry.get_hparam(hparam_set)()
  if extra_hparams:
    hparams.update(extra_hparams)

  cuda = cuda and torch.cuda.is_available()
  if not cuda:
    device = 'cpu'

  if log_dir:
    os.makedirs(log_dir, exist_ok=True)
    if os.listdir(log_dir):
      warnings.warn('Directory "{}" not empty!'.format(log_dir))

    hparams_file_path = os.path.join(log_dir, 'hparams.yaml')
    args_file_path = os.path.join(log_dir, 'args.yaml')

    args = {
        'problem': problem,
        'seed': kwargs.get('seed', None),
    }

    with open(hparams_file_path, 'w') as hparams_file, \
         open(args_file_path, 'w') as args_file:
      yaml.safe_dump(hparams.__dict__, stream=hparams_file,
                     default_flow_style=False)
      yaml.safe_dump(args, stream=args_file,
                     default_flow_style=False)

  problem = problem_cls(hparams, argparse.Namespace(**kwargs),
                        log_dir,
                        device=device,
                        show_progress=progress)

  if resume:
    problem.load_checkpoint(log_dir, epoch=start_epoch)

  problem.run()


@click.command()
@click.argument('problem',
                envvar='PROBLEM', metavar='<problem>',
                required=True)
@click.option('--hparam-set',
              help=('Hyperparameter set name. If not provided, '
                    'first associated used by default.'),
              envvar='HPARAM_SET', metavar='')
@click.option('--extra-hparams',
              help='Comma-separated list of extra key-value pairs.',
              envvar='EXTRA_HPARAMS', metavar='',
              type=ExtraHParamType(), multiple=True)
@click.option('--seed',
              help='Random Seed.',
              envvar='SEED', metavar='', type=int)
@click.option('--progress/--no-progress',
              help='Show/Hide epoch progress.',
              envvar='PROGRESS', metavar='', default=False)
@click.option('--cuda/--no-cuda',
              help='Enable/Disable CUDA.',
              metavar='', default=False)
@click.option('--device',
              help='Device selection.',
              envvar='DEVICE', metavar='', default='cpu')
@click.option('--log-dir',
              help='Directory to store logs.',
              envvar='LOG_DIR', metavar='',
              type=click.Path(file_okay=False,
                              writable=True,
                              resolve_path=True))
@click.option('--log-interval',
              help='Log interval w.r.t epochs.',
              metavar='', default=100, type=int)
@click.option('--eval-interval',
              help='Eval interval w.r.t epochs.',
              metavar='', default=1000, type=int)
@click.option('--num-eval',
              help='Number of evaluations.',
              metavar='', default=10, type=int)
@click.pass_context
def run(ctx, problem,
        hparam_set: str = None,
        progress: bool = False,
        cuda: bool = False,
        device: str = None,
        log_dir: str = None,
        **kwargs):
  """Run Experiments.

  This initializes the Problem class with
  a given Hyperparameter Set. If the hyperparameter
  set is not provided, the first set from the list
  problem's hyperparam sets is used. Arbitrary key
  value pairs can be provided to extend the set from
  command line.
  """

  # Read custom transformation arbitrary CLI key value pairs.
  extra_hparams = ctx.obj.get('extra_hparams')
  kwargs.pop('extra_hparams')

  do_run(problem,
         hparam_set=hparam_set,
         extra_hparams=extra_hparams,
         progress=progress,
         cuda=cuda,
         device=device,
         log_dir=log_dir,
         **kwargs)
