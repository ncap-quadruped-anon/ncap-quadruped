import abc
import dataclasses
import typing as T

import torch
import typing_extensions as TE
from circuits import signal
from utils import collections
from utils.torch import buffers


@dataclasses.dataclass
class Connection:
  source: signal.Signal
  attr: str
  weight: torch.Tensor
  delay: buffers.DelayBuffer
  noise: float
  meta: collections.AttrDict


class Connector:
  def __init__(self):
    self.connections: dict[str, Connection] = {}

  def connect(
    self,
    source: signal.Signal,
    *,
    attr: str = 'value',
    weight: torch.Tensor | float = 1.,
    delay: int = 1,
    noise: float = 0.,
    name: str | None = None,
    **kwargs,
  ):
    if not 0 <= noise <= 1: raise ValueError(f'Expected 0 <= noise <= 1')
    if not delay >= 1: raise ValueError(f'Expected delay >= 1')
    self.connections[name or str(len(self.connections))] = Connection(
      source=source,
      attr=attr,
      weight=torch.as_tensor(weight),
      delay=buffers.DelayBuffer(delay),
      noise=noise,
      meta=collections.AttrDict(kwargs),
    )
    return self

  def __len__(self):
    return len(self.connections)

  def reset(self):
    for c in self.connections.values():
      c.delay.fill()

  def integrate(self) -> torch.Tensor:
    total = torch.zeros(1)
    for c in self.connections.values():
      total = total + c.delay.get()
    return total

  def populate(self):
    for c in self.connections.values():
      x = getattr(c.source, c.attr)
      if c.noise:
        x = torch.nn.functional.dropout(x, p=c.noise)
      x = c.weight * x
      c.delay.step(x)
