# Experiments

A list of hyper-parameters (via CLI arguments) to produce successful models on standard environments.
These arguments are passed to [main.py](./main.py). There's always scope to play around more
and should be easily doable given the flexibility of the program.

## Vanilla A2C on CartPole-v0

```
--env CartPole-v0 --algo a2c --seed 1 --rollout-steps 5 --num-processes 16 --num-total-steps 400000 --gamma 0.99 --alpha 0.5 --beta 0.001 --actor-lr 0.0003 --eval-interval 500
```
