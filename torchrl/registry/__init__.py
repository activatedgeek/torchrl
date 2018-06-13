from torchrl.registry.registry import *  # pylint: disable=wildcard-import


# Trigger decorator imports
def trigger_register():
  # pylint: disable=unused-variable
  import torchrl.registry.hparams


trigger_register()
