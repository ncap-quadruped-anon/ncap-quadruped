import collections
import typing as T

import torch


class DelayBuffer:
  # TODO: Rename to BoundedBuffer? Combine with throttling?
  def __init__(self, delay: int | None = None):
    if delay is not None and delay < 1: raise ValueError(f'Expected delay >= 1')
    self.delay = delay
    self.timesteps = collections.deque(maxlen=delay)
    self.fill()

  def clear(self):
    self.timesteps.clear()

  def fill(self, value: T.Union[torch.Tensor, float] = 0.):
    if self.delay is None: raise RuntimeError('Cannot fill an unbounded buffer')
    self.timesteps.extend(torch.as_tensor(value) for _ in range(self.delay))

  def step(self, elem: T.Union[torch.Tensor, float] = 0.):
    self.timesteps.append(torch.as_tensor(elem))

  def get(self, t: int = 0):
    return self.timesteps[t]

  def __getitem__(self, t: int = 0):
    return self.timesteps[t]

  def __len__(self):
    return len(self.timesteps)

  def __iter__(self):
    return self.timesteps.__iter__()


# ThrottleBuffer
