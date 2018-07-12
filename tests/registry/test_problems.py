from torchrl.registry.problems import HParams


def test_set_hparams():
  hparams = HParams()
  hparams.key = 'value'

  assert hparams.key == 'value'
