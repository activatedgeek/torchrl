import click
import os
from ruamel import yaml

from .run import do_run


@click.command()
@click.argument('log_dir',
                envvar='LOG_DIR', metavar='<log_dir>',
                type=click.Path(file_okay=False,
                                writable=True,
                                resolve_path=True))
@click.option('--progress/--no-progress',
              help='Show/Hide epoch progress.',
              envvar='PROGRESS', metavar='', default=False)
@click.option('--cuda/--no-cuda',
              help='Enable/Disable CUDA.',
              metavar='', default=False)
@click.option('--device',
              help='Device selection.',
              envvar='DEVICE', metavar='', default='cpu')
@click.option('--start-epoch',
              help='Epoch to start with after a load.',
              metavar='', type=int)
@click.option('--log-interval',
              help='Log interval w.r.t epochs.',
              metavar='', default=100, type=int)
@click.option('--eval-interval',
              help='Eval interval w.r.t epochs.',
              metavar='', default=1000, type=int)
@click.option('--num-eval',
              help='Number of evaluations.',
              metavar='', default=10, type=int)
def resume(log_dir, start_epoch,
           progress: bool = False,
           cuda: bool = False,
           device: str = None,
           **kwargs):
  """Resume an experiment from log directory."""

  hparams_file_path = os.path.join(log_dir, 'hparams.yaml')
  args_file_path = os.path.join(log_dir, 'args.yaml')

  with open(hparams_file_path, 'r') as hparams_file, \
    open(args_file_path, 'r') as args_file:
    extra_hparams = yaml.safe_load(hparams_file)
    args = yaml.safe_load(args_file)

  problem = args.pop('problem')
  checkpoint_prefix = args.pop('checkpoint_prefix')
  do_run(problem,
         extra_hparams=extra_hparams,
         progress=progress,
         cuda=cuda,
         device=device,
         log_dir=log_dir,
         resume=True,
         start_epoch=start_epoch,
         checkpoint_prefix=checkpoint_prefix,
         **args,
         **kwargs)
