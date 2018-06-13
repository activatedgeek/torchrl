# Experiments

A list of hyper-parameters (via CLI arguments) to produce successful models on standard environments.
These arguments are passed to [main.py](./experiments/main.py). There's always scope to play around more
and should be easily doable given the flexibility of the program.

All Tensorboard logs are written to `experiments/log` by default.

## DQN on CartPole-v1

```
--env CartPole-v1 --algo dqn --seed 1 --rollout-steps 1 --num-processes 1 --actor-lr 0.0001 --gamma 0.8 \ 
--target-update-interval 5 --eps-min 0.1 --buffer-size 5000 --batch-size 64 --num-total-steps 12000 \
--eval-interval 500
```


## A2C on CartPole-v0

```
--env CartPole-v0 --algo a2c --seed 1 --rollout-steps 5 --num-processes 16 --num-total-steps 1600000 \
--gamma 0.99 --alpha 0.5 --beta 1e-3 --lambda 1.0 --actor-lr 3e-4 --eval-interval 1000
```

## DDPG on Pendulum-v0

```
--env Pendulum-v0 --algo ddpg --seed 1 --rollout-steps 1 --max-episode-steps 500 --num-processes 1 \
--num-total-steps 20000 --gamma 0.99 --buffer-size 1000000 --batch-size 128 --tau 1e-2 --actor-lr 1e-4 \
--critic-lr 1e-3
```

## PPO on Pendulum-v0

```
--env Pendulum-v0 --algo ppo --seed 1 --rollout-steps 20 --num-processes 16 --num-total-steps 5000000 \
--gamma 0.99 --alpha 0.5 --beta 1e-3 --lambda 0.95 --clip-ratio 0.2 --actor-lr 3e-4 --ppo-epochs 4 \
--batch-size 5 --eval-interval 500
```

# Issues

Feel free to open up issues. I'm making these interfaces better to allow faster experimentation with new
ideas.
