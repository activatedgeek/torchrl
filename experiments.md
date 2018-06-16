# Experiments

A list of hyper-parameters (via CLI arguments) to produce successful models on standard environments.
There's always scope to play around more and should be easily doable given the flexibility of the program.

All Tensorboard logs are written to `$(pwd)/log` by default.

## DQN on CartPole-v1

```
$ torchrl --problem=dqn-cartpole-v1 --hparam-set=dqn-cartpole --usr-dirs=experiments
```


## A2C on CartPole-v0

```
$ torchrl --problem=a2c-cartpole-v0 --hparam-set=a2c-cartpole --usr-dirs=experiments
```

## DDPG on Pendulum-v0


```
$ torchrl --problem=ddpg-pendulum-v0 --hparam-set=ddpg-pendulum --usr-dirs=experiments
```

## PPO on Pendulum-v0

```
$ torchrl --problem=ppo-pendulum-v0 --hparam-set=ppo-pendulum --usr-dirs=experiments
```

# Issues

Feel free to open up issues. I'm making these interfaces better to allow faster experimentation with new
ideas.
