import abc
import typing as T

import numpy as np
import torch
from circuits import connector, signal
from utils.torch import activations


def _repr_param(x: float | torch.Tensor) -> str:
  if isinstance(x, torch.Tensor):
    name = x.__class__.__name__
    if len(x.shape) == 0:
      return f'{name}({x.item():.3g})'
    else:
      return f'{name}{list(x.shape)}'
  elif isinstance(x, float):
    return f'{x:.3g}'
  else:
    return str(x)


class Unit(signal.Signal, abc.ABC):
  def __init__(self, **kwargs):
    super().__init__(**kwargs)

  def step(self):
    pass

  def populate(self):
    pass


class Basic(Unit):
  def __init__(
    self,
    activation=activations.unitrelu,
    bias: float | torch.Tensor = 0,
    voltage_time: float = 50,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.activation = activation

    assert -1 <= bias <= 1
    assert 0 < voltage_time

    self.bias = bias
    self.voltage_time = voltage_time

    self.synapses = connector.Connector()

    self.v: torch.Tensor  # Voltage variable
    self.dt: float  # Timestep

  def synapse(self, source: signal.Signal, **kwargs):
    self.synapses.connect(source, **kwargs)
    return self

  def init(self, cfg):
    self.dt = cfg.timestep

  def reset(self):
    self.synapses.reset()
    self.v = torch.zeros(1)
    self.value = torch.zeros(1)

  def step(self):
    v = self.v
    y = self.activation

    z = torch.clamp(self.synapses.integrate() + self.bias, min=-1, max=1)

    # Forward-Euler discretization.
    # dv = z - v
    # self.v = v + 4 * dv / self.voltage_time * self.dt

    # Backward-Euler discretization.
    kv = 4 / self.voltage_time * self.dt
    self.v = (v + z * kv) / (1 + kv)

    self.value = y(self.v)

  def populate(self):
    self.synapses.populate()

  def extra_repr(self) -> str:
    cfg = dict(
      B=self.bias,
      Tv=self.voltage_time,
    )
    return ', '.join([f'{k}={_repr_param(v)}' for k, v in cfg.items()])


class Adaptor(Unit):
  def __init__(
    self,
    activation=activations.unitrelu,
    bias: float | torch.Tensor = 0,
    adaptation: float = 0,
    adaptation_time: float = 1000,
    voltage_time: float = 50,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.activation = activation

    assert -1 <= bias <= 1
    assert -1 <= adaptation <= 1
    assert not (adaptation == 1 and bias == 0)
    assert 0 < adaptation_time
    assert 0 < voltage_time

    self.bias = bias
    self.adaptation = adaptation
    self.adaptation_time = adaptation_time
    self.voltage_time = voltage_time

    self.synapses = connector.Connector()

    self.v: torch.Tensor  # Voltage variable
    self.a: torch.Tensor  # Adaptation variable
    self.dt: float  # Timestep

  def synapse(self, source: signal.Signal, **kwargs):
    self.synapses.connect(source, **kwargs)
    return self

  def init(self, cfg):
    self.dt = cfg.timestep

  def reset(self):
    self.synapses.reset()
    self.v = torch.zeros(1)
    self.a = torch.zeros(1)
    self.value = torch.zeros(1)

  def step(self):
    v = self.v
    a = self.a
    y = self.activation

    z = torch.clamp(self.synapses.integrate() + self.bias, min=-1, max=1)

    # Forward-Euler discretization.
    # dv = z - v + a
    # if self.adaptation > 0:
    #   # Positive adaptation -> facilitation.
    #   da = self.adaptation * y(v) * (1 - torch.clamp(z, 0, 1)) - a
    # else:
    #   # Negative adaptation -> depression.
    #   da = self.adaptation * torch.sign(y(v)) * torch.clamp(z, 0, 1) - a
    # self.v = v + 4 * dv / self.voltage_time * self.dt
    # self.a = a + 4 * da / self.adaptation_time * self.dt

    # Backward-Euler discretization.
    kv = 4 / self.voltage_time * self.dt
    self.v = (v + (z + a) * kv) / (1 + kv)
    if self.adaptation > 0:
      # Positive adaptation -> facilitation.
      x = self.adaptation * y(self.v) * (1 - torch.clamp(z, 0, 1))
    else:
      # Negative adaptation -> depression.
      x = self.adaptation * torch.sign(y(self.v)) * torch.clamp(z, 0, 1)
    ka = 4 / self.adaptation_time * self.dt
    self.a = (a + x * ka) / (1 + ka)

    self.value = y(self.v)

  def populate(self):
    self.synapses.populate()

  def extra_repr(self) -> str:
    cfg = dict(
      B=self.bias,
      A=self.adaptation,
      Ta=self.adaptation_time,
      Tv=self.voltage_time,
    )
    return ', '.join([f'{k}={_repr_param(v)}' for k, v in cfg.items()])


def linear(start: float, end: float, z: torch.Tensor) -> torch.Tensor:
  return start * (1 - z) + end * z


def exponential(start: float, end: float, z: torch.Tensor) -> torch.Tensor:
  assert start > 0
  assert end > 0
  return (start**(1 - z)) * (end**z)


def halfconvex(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
  return (v > 0) * (1 + torch.sqrt(a)) / 2


def halflinear(v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
  return (v > 0) * (a + 1) / 2


class Oscillator(Unit):
  def __init__(
    self,
    activation=halflinear,
    bias: float | torch.Tensor = 0,
    adaptation_time: float = 2000,
    active_time: float = 500,
    quiet_time: float = 1000,
    active_scale: float = 0.5,
    quiet_scale: float = 0.1,
    tonic_threshold: float = 1.0,
    **kwargs,
  ):
    super().__init__(**kwargs)
    self.activation = activation

    assert -1 <= bias <= 1
    assert 0 < adaptation_time
    assert 0 < active_time
    assert 0 < quiet_time
    assert 0 < active_scale
    assert 0 < quiet_scale

    self.bias = bias
    self.adaptation_time = adaptation_time
    self.active_time = active_time
    self.quiet_time = quiet_time
    self.active_scale = active_scale
    self.quiet_scale = quiet_scale
    self.tonic_threshold = tonic_threshold

    ka = 4 * active_time / adaptation_time
    kq = 4 * quiet_time / adaptation_time
    self.active_bound0 = (1 - np.exp(kq)) / (1 - np.exp(ka + kq))
    self.quiet_bound0 = self.active_bound0 * np.exp(ka)
    self.active_bound1 = (1 - np.exp(kq * quiet_scale
                                    )) / (1 - np.exp(ka * active_scale + kq * quiet_scale))
    self.quiet_bound1 = self.active_bound1 * np.exp(ka * active_scale)

    self.synapses = connector.Connector()

    self.v: torch.Tensor  # Voltage variable
    self.a: torch.Tensor  # Adaptation variable
    self.dt: float  # Timestep

  def synapse(self, source: signal.Signal, **kwargs):
    self.synapses.connect(source, **kwargs)
    return self

  def init(self, cfg):
    self.dt = cfg.timestep

  def reset(self):
    self.synapses.reset()
    self.v = -torch.ones(1)
    self.a = torch.rand(1) * (self.quiet_bound0 - self.active_bound0) + self.active_bound0
    self.value = torch.zeros(1)

  def step(self):
    v = self.v
    a = self.a
    y = self.activation

    x = self.synapses.integrate() + self.bias
    z = torch.clamp(x, min=0, max=1)

    # Forward-Euler discretization.
    # da = (v < 0) * (1 - a) + (v > 0) * (0 - a)
    # a = a + 4 * da / self.adaptation_time * self.dt
    # a = torch.clamp(a, min=0, max=1)

    # Backward-Euler discretization.
    k = 4 / self.adaptation_time * self.dt
    a = (v < 0) * (a + k) / (1 + k) + (v > 0) * a / (1 + k)
    a = torch.clamp(a, min=0, max=1)

    active_bound = linear(self.active_bound0, self.active_bound1, z)
    quiet_bound = linear(self.quiet_bound0, self.quiet_bound1, z)
    active_to_quiet = (a <= active_bound) & (x <= self.tonic_threshold)
    quiet_to_active = (a >= quiet_bound) & (x >= 0)
    v = v.clone()
    v[active_to_quiet] = -1
    v[quiet_to_active] = +1

    self.v = v
    self.a = a
    self.value = y(self.v, self.a)

  def populate(self):
    self.synapses.populate()

  def extra_repr(self) -> str:
    cfg = dict(
      B=self.bias,
      T=self.adaptation_time,
      Ta=self.active_time,
      Tq=self.quiet_time,
      Ka=self.active_scale,
      Kq=self.quiet_scale,
    )
    return ', '.join([f'{k}={_repr_param(v)}' for k, v in cfg.items()])
