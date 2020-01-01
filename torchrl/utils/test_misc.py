import os
import sys
from torchrl.utils.misc import import_usr_dir, to_camel_case


def test_import_usr_dir():
  usr_dir = os.path.dirname(__file__)
  import_usr_dir(usr_dir)

  assert os.path.dirname(usr_dir) not in sys.path


def test_camel_case():
  assert to_camel_case('A2C4') == 'a2c4'
  assert to_camel_case('ABC') == 'abc'
  assert to_camel_case('DoesThisWork') == 'does_this_work'
  assert to_camel_case('doesThisAlsoWork') == 'does_this_also_work'
