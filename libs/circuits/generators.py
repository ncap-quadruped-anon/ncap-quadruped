import abc
import typing as T

import numpy as np
import torch
from circuits import signal


class Generator(signal.Signal, abc.ABC):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def step(self):
    pass


class PeriodicGenerator(Generator):
  def __init__(
    self,
    pattern: torch.Tensor,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.pattern: torch.Tensor = pattern

  def reset(self):
    self.value = self.pattern[..., 0]
    self.index = 0

  def step(self):
    self.index = (self.index + 1) % self.pattern.shape[-1]
    self.value = self.pattern[..., self.index, None]


class SquareGenerator(PeriodicGenerator):
  def __init__(
    self,
    period: int = 2,
    duration: int = 1,
    phase: int = 0,
    low: float = 0.,
    high: float = 1.,
    **kwargs,
  ):
    if not period >= 1: raise ValueError(f'Expected period >= 1')
    if not duration >= 0: raise ValueError(f'Expected duration >= 0')
    pattern = torch.full((period,), low)
    pattern[phase:phase + duration] = high
    super().__init__(pattern, **kwargs)


class SineGenerator(PeriodicGenerator):
  def __init__(
    self,
    period: int = 4,
    amplitude: float = 1.,
    phase: float = 0.,
    shift: float = 0.,
    **kwargs,
  ):
    if period < 1: raise ValueError(f'Expected period >= 1')
    pattern = amplitude * torch.sin(2 * np.pi / period * (torch.arange(0., period) - phase)) + shift
    super().__init__(pattern, **kwargs)
