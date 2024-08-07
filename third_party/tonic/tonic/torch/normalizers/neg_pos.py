import typing as T
import numpy as np
import torch


class NegPos(torch.nn.Module):
  def __init__(
    self,
    min: int | float | np.ndarray = 0,
    mid: int | float | np.ndarray = 0,
    max: int | float | np.ndarray = 0,
    freeze: bool = False,
    shape: tuple[int, ...] | None = None,
  ):
    super().__init__()
    self._min = np.array(min)
    self._mid = np.array(mid)
    self._max = np.array(max)
    self._freeze = freeze
    if shape:
      self.initialize(shape)

  def initialize(self, shape: tuple[int, ...]):
    # Convert _min, _mid, _max to torch.Tensor with shape.
    self._min = np.broadcast_to(self._min, shape).copy()
    self._mid = np.broadcast_to(self._mid, shape).copy()
    self._max = np.broadcast_to(self._max, shape).copy()
    self._min = np.minimum(self._min, self._mid)
    self._max = np.maximum(self._max, self._mid)
    self.min = torch.nn.Parameter(torch.tensor(self._min, dtype=torch.float32), requires_grad=False)
    self.mid = torch.nn.Parameter(torch.tensor(self._mid, dtype=torch.float32), requires_grad=False)
    self.max = torch.nn.Parameter(torch.tensor(self._max, dtype=torch.float32), requires_grad=False)

  def forward(self, val: torch.Tensor):
    with torch.no_grad():
      # Convert [min, _mid, max] to [-1, 0, 1]:
      # val' = (-1 + (val - min) / (mid - min)) if val < mid else (0 + (val - mid) / (max - mid))
      val = torch.clamp(val, self.min, self.max)
      neg = (val < self.mid).float() * ((val - self.min) / (self.mid - self.min) - 1)
      pos = (val >= self.mid).float() * (val - self.mid) / (self.max - self.mid)
      val = torch.nan_to_num(neg, 0, 0, 0) + torch.nan_to_num(pos, 0, 0, 0)
    return val

  def unnormalize(self, val: torch.Tensor):
    with torch.no_grad():
      # Convert [-1, 0, 1] to [min, mid, max]:
      # val' = min + (val + 1) * (mid - min) if val < 0 else mid + val * (max - mid)
      val = torch.clamp(val, -1, 1)
      val = ((val < 0).float() * (self.min + (val + 1) * (self.mid - self.min)) +
             (val >= 0).float() * (self.mid + val * (self.max - self.mid)))
    return val

  def record(self, values: T.Iterable[np.ndarray]):
    if self._freeze: return
    for val in values:
      self._min = np.minimum(self._min, val)
      self._max = np.maximum(self._max, val)

  def update(self):
    if self._freeze: return
    self.min.data.copy_(torch.tensor(self._min, dtype=torch.float32))
    self.max.data.copy_(torch.tensor(self._max, dtype=torch.float32))

  def get_new_stats(self) -> tuple[np.ndarray, np.ndarray]:
    return (self._min, self._max)

  def record_new_stats(self, new_stats: tuple[np.ndarray, np.ndarray]):
    if self._freeze: return
    min, max = new_stats
    self._min = np.minimum(self._min, min)
    self._max = np.maximum(self._max, max)

  def get_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (self._min, self._mid, self._max)

  def set_state(self, state: tuple[np.ndarray, np.ndarray, np.ndarray]):
    self._min, self._mid, self._max = state
    self.min.data.copy_(torch.tensor(self._min, dtype=torch.float32))
    self.mid.data.copy_(torch.tensor(self._mid, dtype=torch.float32))
    self.max.data.copy_(torch.tensor(self._max, dtype=torch.float32))
