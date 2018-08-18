import click
import json
from ruamel import yaml
import sys

from ... import registry


@click.group(name='list')
@click.option('--output', '-o',
              help="Output format.",
              type=click.Choice(['default', 'json', 'yaml']),
              default='default')
@click.pass_context
def group_list(ctx, output):
  """List Resources."""
  ctx.obj['output'] = output


@group_list.command(help=("List all registered Problems with "
                          "corresponding hyper-parameter sets."))
@click.pass_context
def problems(ctx):
  if ctx.obj['output'] == 'json':
    click.echo(json.dumps(registry.list_problem_hparams(), indent=2))
  elif ctx.obj['output'] == 'yaml':
    yaml.dump(registry.list_problem_hparams(), sys.stdout,
              default_flow_style=False)
  else:
    for problem, p_hparams in registry.list_problem_hparams().items():
      click.echo('{}:'.format(problem))
      for hparam in p_hparams:
        click.echo('\t{}'.format(hparam))
      click.echo()


@group_list.command(help="List all registered hyper-parameter sets.")
@click.pass_context
def hparams(ctx):
  if ctx.obj['output'] == 'json':
    click.echo(json.dumps(registry.list_hparams(), indent=2))
  elif ctx.obj['output'] == 'yaml':
    yaml.dump(registry.list_hparams(), sys.stdout,
              default_flow_style=False)
  else:
    for hparam in registry.list_hparams():
      click.echo(hparam)
