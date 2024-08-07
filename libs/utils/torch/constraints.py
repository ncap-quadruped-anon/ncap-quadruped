import typing as T

import torch

ConstraintBounds = tuple[int | float | None, int | float | None]
ConstraintFn = T.Callable[[torch.nn.Parameter], torch.nn.Parameter]


class ConstrainedParameter(torch.nn.Parameter):
  def __new__(
    cls,
    data: torch.Tensor | None = None,
    constraint: ConstraintBounds | ConstraintFn | None = None,
    requires_grad: bool = True,
  ):
    self = super().__new__(cls, data=data, requires_grad=requires_grad)

    if isinstance(constraint, tuple):
      if not len(constraint) == 2 or (constraint[0] is None and constraint[1] is None):
        raise ValueError(f'Invalid constraint bounds: {constraint}')
      lower, upper = constraint

      def clamp_(p):
        return p.clamp_(min=lower, max=upper)

      constraint = clamp_
    self.constraint = constraint

    return self

  def __deepcopy__(self, memo):
    if id(self) in memo:
      return memo[id(self)]
    else:
      result = type(self)(
        self.data.clone(memory_format=torch.preserve_format), self.constraint, self.requires_grad
      )
      memo[id(self)] = result
      return result

  def __repr__(self):
    return 'Constrained' + super().__repr__()

  @torch.no_grad()
  def constrain(self):
    if self.constraint is not None:
      self.data = self.constraint(self.data)
    return self


class Constrainer:
  def __init__(self, params):
    self.params = list(param for param in params if isinstance(param, ConstrainedParameter))

  def constrain(self):
    for param in self.params:
      param.constrain()
