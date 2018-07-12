# pylint: disable=redefined-outer-name

import random
import pytest
import torchrl.registry.registry as registry
from torchrl.registry import HParams


@pytest.fixture(scope='function')
def hparam_set_fixture():
  hparam_set_id = 'test-custom-hparams-{}'.format(random.randint(1, 100))
  random_int = random.randint(1, 100)

  @registry.register_hparam(hparam_set_id)
  def custom_hparams():  # pylint: disable=unused-variable
    hparams = HParams()
    hparams.custom_value = random_int

    return hparams

  yield hparam_set_id, random_int

  registry.remove_hparam(hparam_set_id)


def test_register_hparam(hparam_set_fixture: tuple):
  hparam_set_id, random_int = hparam_set_fixture

  hparams_list = registry.list_hparams()

  assert hparam_set_id in hparams_list

  hparams = registry.get_hparam(hparam_set_id)()

  assert isinstance(hparams, HParams)
  assert hparams.custom_value == random_int


def test_hparams_registry_list():
  assert isinstance(registry.list_hparams(), list)


def test_problems_registry_list():
  assert isinstance(registry.list_problems(), list)
