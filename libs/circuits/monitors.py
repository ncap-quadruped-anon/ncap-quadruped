import abc
import dataclasses
import typing as T

from circuits import nodes
from utils import collections
from utils.torch import buffers


class Monitor(nodes.PrimitiveNode, abc.ABC):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def populate(self):
    pass


@dataclasses.dataclass
class TimeseriesMonitorSource:
  source: T.Any
  attr: str
  buffer: buffers.DelayBuffer
  meta: collections.AttrDict


class TimeseriesMonitor(Monitor):
  def __init__(self, timesteps: int | None = 20, **kwargs):
    super().__init__(**kwargs)
    self.timesteps = timesteps
    self.sources: T.List[TimeseriesMonitorSource] = []

  def connect(self, source: T.Any, attr: str = 'value', **kwargs):
    if not hasattr(source, attr):
      raise AttributeError(f"{type(source).__name__} source does not have attribute '{attr}'")
    self.sources.append(
      TimeseriesMonitorSource(
        source=source,
        attr=attr,
        buffer=buffers.DelayBuffer(self.timesteps),
        meta=collections.AttrDict(kwargs),
      )
    )
    return self

  def reset(self):
    for source in self.sources:
      source.buffer.clear()

  def populate(self):
    for source in self.sources:
      value = getattr(source.source, source.attr)
      source.buffer.step(value)
