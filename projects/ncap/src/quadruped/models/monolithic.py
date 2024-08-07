import typing as T

import gym.spaces
import gym.spaces.utils
import numpy as np
import torch
from tonic.torch import models, normalizers


def monolithic_ff_d4pg(
  actor_hidden_sizes: T.Sequence[int] | None = (256, 256),
  actor_hidden_activation: T.Callable = torch.nn.ReLU,
  actor_hidden_init: T.Callable | None = None,
  actor_head_activation: T.Callable = torch.nn.Tanh,
  actor_head_bias: bool = True,
  actor_head_init: T.Callable | None = None,
  critic_hidden_sizes: T.Sequence[int] | None = (256, 256),
  critic_hidden_activation: T.Callable = torch.nn.ReLU,
  critic_hidden_init: T.Callable | None = None,
  critic_value_distribution: tuple[float, float, int] = (0., 150., 51),
  critic_head_init: T.Callable | None = None,
  observation_normalizer: bool = True,
):
  return models.ActorCriticWithTargets(
    actor=models.Actor(
      encoder=models.ObservationEncoder(),
      torso=models.MLP(
        sizes=actor_hidden_sizes,
        activation=actor_hidden_activation,
        fn=actor_hidden_init,
      ) if actor_hidden_sizes else None,
      head=models.DeterministicPolicyHead(
        activation=actor_head_activation,  # type: ignore
        bias=actor_head_bias,
        fn=actor_head_init,
      ),
    ),
    critic=models.Critic(
      encoder=models.ObservationActionEncoder(),
      torso=models.MLP(
        sizes=critic_hidden_sizes,
        activation=critic_hidden_activation,
        fn=critic_hidden_init,
      ) if critic_hidden_sizes else None,
      head=models.DistributionalValueHead(
        *critic_value_distribution,
        fn=critic_head_init,
      ),
    ),
    observation_normalizer=normalizers.MeanStd() if observation_normalizer else None,
  )


def monolithic_ff_ppo(
  actor_hidden_sizes: T.Sequence[int] | None = (256, 256),
  actor_hidden_activation: T.Callable = torch.nn.ReLU,
  actor_hidden_init: T.Callable | None = None,
  actor_head_activation: T.Callable = torch.nn.Tanh,
  actor_head_init: T.Callable | None = None,
  critic_hidden_sizes: T.Sequence[int] | None = (256, 256),
  critic_hidden_activation: T.Callable = torch.nn.ReLU,
  critic_hidden_init: T.Callable | None = None,
  critic_head_init: T.Callable | None = None,
  observation_normalizer: bool = True,
):
  return models.ActorCritic(
    actor=models.Actor(
      encoder=models.ObservationEncoder(),
      torso=models.MLP(
        sizes=actor_hidden_sizes,
        activation=actor_hidden_activation,
        fn=actor_hidden_init,
      ) if actor_hidden_sizes else None,
      head=models.DetachedScaleGaussianPolicyHead(
        loc_activation=actor_head_activation,  # type: ignore
        loc_fn=actor_head_init,
      ),
    ),
    critic=models.Critic(
      encoder=models.ObservationEncoder(),
      torso=models.MLP(
        sizes=critic_hidden_sizes,
        activation=critic_hidden_activation,
        fn=critic_hidden_init,
      ) if critic_hidden_sizes else None,
      head=models.ValueHead(fn=critic_head_init),
    ),
    observation_normalizer=normalizers.MeanStd() if observation_normalizer else None,
  )


def monolithic_ff_es(
  actor_hidden_sizes: T.Sequence[int] | None = (256, 256),
  actor_hidden_activation: T.Callable = torch.nn.ReLU,
  actor_hidden_init: T.Callable | None = None,
  actor_head_activation: T.Callable = torch.nn.Tanh,
  actor_head_bias: bool = True,
  actor_head_init: T.Callable | None = None,
  observation_normalizer: bool = True,
):
  return models.ActorOnly(
    actor=models.Actor(
      encoder=models.ObservationEncoder(),
      torso=models.MLP(
        sizes=actor_hidden_sizes,
        activation=actor_hidden_activation,
        fn=actor_hidden_init,
      ) if actor_hidden_sizes else None,
      head=models.DeterministicPolicyHead(
        activation=actor_head_activation,  # type: ignore
        bias=actor_head_bias,
        fn=actor_head_init,
      ),
    ),
    observation_normalizer=normalizers.MeanStd() if observation_normalizer else None,
  )
