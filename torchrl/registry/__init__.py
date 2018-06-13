from torchrl.registry.registry import *


# Trigger decorator imports
def trigger_register():
  import torchrl.registry.hparams


trigger_register()
