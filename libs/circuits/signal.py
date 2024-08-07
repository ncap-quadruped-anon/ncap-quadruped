import typing as T

import torch
from circuits import nodes


class Signal(nodes.PrimitiveNode):
  def __init__(self, value: T.Union[torch.Tensor, float] = 0., **kwargs):
    super().__init__(**kwargs)
    self.value: torch.Tensor = torch.as_tensor(value)  # Allow pointers to the same tensor.
