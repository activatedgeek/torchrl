# Experiments

## Example Usage

Here's an example usage of the A2C Cartpole experiment

```python
from experiments.a2c.cartpole_v0 import A2CCartpole

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

args=dict(
    seed=1,
    log_interval=1000,
    eval_interval=1000,
    num_eval=1,
)

hparams = A2CCartpole.default_hparams()

a2c_cartpole = A2CCartpole(
    hparams,
    argparse.Namespace(**args),
    'log/a2c_cartpole',
    device=device,
    show_progress=True,
)

a2c_cartpole.run()
```

Other experiments can be run similarly.
