# Experiments

A list of hyper-parameters (via CLI arguments) to produce successful models on standard environments.
These arguments are passed to [main.py](./experiments/main.py). There's always scope to play around more
and should be easily doable given the flexibility of the program.

All Tensorboard logs are written to `experiments/log` by default.

## A2C on CartPole-v0

```
--env CartPole-v0 --algo a2c --seed 1 --rollout-steps 5 --num-processes 16 --num-total-steps 400000 \
--gamma 0.99 --alpha 0.5 --beta 0.001 --actor-lr 0.0003 --eval-interval 500
```

## DDPG on Pendulum-v0

```
--env Pendulum-v0 --algo ddpg --seed 1 --rollout-steps 1 --max-episode-steps 500 --num-processes 1 \
--num-total-steps 20000 --gamma 0.99 --buffer-size 1000000 --batch-size 128 --tau 1e-2 --actor-lr 1e-4 \
--critic-lr 1e-3
```

# Issues

Feel free to open up issues. I'm making these interfaces better to allow faster experimentation with new
ideas.
