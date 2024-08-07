import typing as T

import gym
import gym.spaces
import gym.spaces.utils
import torch
from tonic.torch import normalizers


def meanstd_builder(key: str) -> normalizers.MeanStd:
  return normalizers.MeanStd()


def posneg_builder(key: str) -> normalizers.NegPos:
  return normalizers.NegPos()


Normalizer = normalizers.MeanStd | normalizers.NegPos


class UnflatNormalizer(torch.nn.Module):
  def __init__(
    self,
    normalizer_builder: T.Mapping[str, Normalizer] | T.Callable[[str], Normalizer] = posneg_builder,
  ):
    super().__init__()
    self.normalizer_builder = normalizer_builder
    self.normalizers = torch.nn.ModuleDict()

  def initialize(self, observation_space: gym.spaces.Dict):
    assert isinstance(observation_space, gym.spaces.Dict)
    for key, o in observation_space.spaces.items():
      assert isinstance(o, gym.spaces.Box)
      if callable(self.normalizer_builder):
        normalizer = self.normalizer_builder(key)
      else:
        normalizer = self.normalizer_builder[key]
      normalizer.initialize(o.shape)
      self.normalizers[key] = normalizer

  def forward(self, observation: dict[str, torch.Tensor]):
    observation = observation.copy()
    for key, o in observation.items():
      if key in self.normalizers:
        observation[key] = self.normalizers[key](o)
    return observation

  def unnormalize(self, observation: dict[str, torch.Tensor]):
    observation = observation.copy()
    for key, o in observation.items():
      if key in self.normalizers:
        observation[key] = self.normalizers[key].unnormalize(o)  # type: ignore
    return observation

  def record(self, observations: T.Iterable[dict[str, torch.Tensor]]):
    for observation in observations:
      for key, val in observation.items():
        if key in self.normalizers:
          self.normalizers[key].record(val)  # type: ignore

  def update(self):
    for normalizer in self.normalizers.values():
      normalizer.update()  # type: ignore

  def get_new_stats(self):
    return {
      key:
        normalizer.get_new_stats()  # type: ignore
      for key,
      normalizer in self.normalizers.items()
    }

  def record_new_stats(self, new_stats):
    for key, stats in new_stats.items():
      self.normalizers[key].record_new_stats(stats)  # type: ignore

  def get_state(self):
    return {
      key:
        normalizer.get_state()  # type: ignore
      for key, normalizer in self.normalizers.items()
    }

  def set_state(self, state):
    for key, s in state.items():
      self.normalizers[key].set_state(s)  # type: ignore


class UnflatMeanStd(torch.nn.Module):
  def __init__(
    self,
    normalizer_builder: T.Callable[[str], normalizers.MeanStd] = meanstd_builder,
  ):
    super().__init__()
    self.normalizer_builder = normalizer_builder
    self.normalizers = torch.nn.ModuleDict()

  def initialize(self, observation_space: gym.spaces.Dict):
    assert isinstance(observation_space, gym.spaces.Dict)
    for key, o in observation_space.spaces.items():
      assert isinstance(o, gym.spaces.Box)
      normalizer = self.normalizer_builder(key)
      normalizer.initialize(o.shape)
      self.normalizers[key] = normalizer

  def forward(self, observation: dict[str, torch.Tensor]):
    observation = observation.copy()
    for key, o in observation.items():
      if key in self.normalizers:
        observation[key] = self.normalizers[key](o)
    return observation

  def unnormalize(self, observation: dict[str, torch.Tensor]):
    observation = observation.copy()
    for key, o in observation.items():
      if key in self.normalizers:
        observation[key] = self.normalizers[key].unnormalize(o)  # type: ignore
    return observation

  def record(self, observations: T.Iterable[dict[str, torch.Tensor]]):
    for observation in observations:
      for key, val in observation.items():
        if key in self.normalizers:
          self.normalizers[key].record(val)  # type: ignore

  def update(self):
    for normalizer in self.normalizers.values():
      normalizer.update()  # type: ignore

  def get_new_stats(self):
    return {
      key:
        normalizer.get_new_stats()  # type: ignore
      for key,
      normalizer in self.normalizers.items()
    }

  def record_new_stats(self, new_stats):
    for key, stats in new_stats.items():
      self.normalizers[key].record_new_stats(stats)  # type: ignore

  def get_state(self):
    return {
      key:
        normalizer.get_state()  # type: ignore
      for key, normalizer in self.normalizers.items()
    }

  def set_state(self, state):
    for key, s in state.items():
      self.normalizers[key].set_state(s)  # type: ignore


class UnflatActorOnly(torch.nn.Module):
  def __init__(self, actor, observation_normalizer: UnflatMeanStd | UnflatNormalizer | None = None):
    super().__init__()
    self.actor = actor
    self.observation_normalizer = observation_normalizer

  def initialize(self, observation_space, action_space):
    assert isinstance(observation_space, gym.spaces.Dict)
    if self.observation_normalizer:
      self.observation_normalizer.initialize(observation_space)  # Pass space not shape.
    self.actor.initialize(observation_space, action_space, self.observation_normalizer)

  def reset(self):
    if hasattr(self.actor, 'reset'):
      self.actor.reset()

  def forward(self, *inputs):
    return self.actor(*inputs)


class UnflatActor(torch.nn.Module):
  def __init__(
    self,
    actor: torch.nn.Module,
    observation_unflat: bool = True,
    observation_normalizer: bool = True,
    action_unflat: bool = True,
    action_distribution: T.Callable[[torch.Tensor], torch.distributions.Distribution] | None = None,
  ):
    super().__init__()
    self.actor = actor
    self.observation_unflat = observation_unflat
    self.observation_normalizer = observation_normalizer  # type: ignore
    self.action_unflat = action_unflat
    self.action_distribution = action_distribution

  def initialize(self, observation_space, action_space, observation_normalizer=None):
    # Must use spaces generated by `tonic.enviornments.wrappers.FlattenObservationAction` in order
    # to access the original spaces, which are used to convert between the flat reprs that Tonic
    # requires and the unflat reprs that facilitate named access to specific keys.
    assert hasattr(observation_space, 'original')
    assert hasattr(action_space, 'original')
    self.observation_space_unflat = observation_space.original
    self.action_space_unflat = action_space.original
    self.observation_normalizer: torch.nn.Module | None = (
      observation_normalizer if self.observation_normalizer else None
    )
    self.actor.initialize(  # type: ignore
      self.observation_space_unflat if self.observation_unflat else observation_space,
      self.action_space_unflat if self.action_unflat else action_space,
      self.observation_normalizer,
    )

  def reset(self):
    if hasattr(self.actor, 'reset'):
      self.actor.reset()

  def forward(self, observations: torch.Tensor) -> torch.Tensor | torch.distributions.Distribution:
    """
    Args:
      observations: Flat observations, shape (batch_size, observation_size).
    Returns:
      actions: Flat actions, shape (batch_size, action_size).
    """
    if self.observation_normalizer:
      observations = self.observation_normalizer(observations)
    if self.observation_unflat:
      observations = gym.spaces.utils.unflatten(self.observation_space_unflat, observations)
    actions = self.actor(observations)
    if self.action_unflat:
      actions = gym.spaces.utils.flatten(self.action_space_unflat, actions)
    if self.action_distribution:
      actions = self.action_distribution(actions)
    return actions
