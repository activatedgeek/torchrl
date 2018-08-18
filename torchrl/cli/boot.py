import click

from ..utils.misc import import_usr_dir

from .commands.list import group_list
from .commands.run import run
from .commands.resume import resume


usr_dirs_help = ("Path to user module(s) which can register "
                 "problems or hparams. Multiple uses of this "
                 "flag are allowed. Optionally, use environment "
                 "variable USR_DIRS separated by a ':'. "
                 "e.g. USR_DIRS=/path/a:/path/b")
@click.group()
@click.option('--usr-dirs',
              help=usr_dirs_help,
              envvar='USR_DIRS',
              metavar='',
              multiple=True,
              type=click.Path(file_okay=False))
def cli(usr_dirs):
  """TorchRL CLI."""
  for usr_dir in usr_dirs:
    import_usr_dir(usr_dir)


cli.add_command(group_list)
cli.add_command(run)
cli.add_command(resume)


def main():
  cli(obj={  # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
      'extra_hparams': {},
      'resume': False,
      'start_epoch': None
  })


if __name__ == '__main__':
  main()
