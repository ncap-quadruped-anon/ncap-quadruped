import collections
import typing as T

import numpy.typing as npt
import torch
import utils.string as uts

ParameterFactory = T.Callable[[], torch.nn.Parameter]
TensorLike = float | npt.ArrayLike | npt.NDArray | torch.Tensor


def tensorlike_parameter_factory(data: TensorLike, requires_grad: bool = True) -> ParameterFactory:
  def factory():
    return torch.nn.Parameter(torch.tensor(data), requires_grad=requires_grad)

  return factory


def _repr_param(x: float | torch.Tensor | torch.nn.Parameter) -> str:
  if isinstance(x, torch.nn.Parameter):
    name = x.__class__.__name__
    if len(x.shape) == 0:
      return f'{name}({x.item():.3g}, requires_grad={x.requires_grad})'
    else:
      return f'{name}(shape={list(x.shape)}, requires_grad={x.requires_grad})'
  elif isinstance(x, torch.Tensor):
    name = x.__class__.__name__
    if len(x.shape) == 0:
      return f'{name}({x.item():.3g})'
    else:
      return f'{name}{list(x.shape)}'
  elif isinstance(x, float):
    return f'{x:.3g}'
  else:
    return str(x)


class ParameterManager(torch.nn.ParameterDict):
  def __init__(self, **kwargs):
    # Maps original -> [alias1, alias2, ...].
    self._shared: dict[str, list[str]] = collections.defaultdict(list)
    self._factories: dict[str, ParameterFactory] = {}
    super().__init__(**kwargs)

  def config(
    self,
    original: str,
    share: str | T.Iterable[str] | None = None,
    factory: TensorLike | ParameterFactory | None = None,
    requires_grad: bool = True,
  ):
    """Convenience method to simultaneously call `share()` and `factory()`."""
    if share is not None:
      self.share(original, share)
    if factory is not None:
      if not callable(factory):
        factory = tensorlike_parameter_factory(factory, requires_grad=requires_grad)
      self.factory(original, factory)
      if share is not None:
        self.factory(share, factory)

  def share(self, original: str, aliases: str | T.Iterable[str]):
    """Register names matching any patterns in `aliases` to point to the `original` name."""
    if isinstance(aliases, str):
      self._shared[original].append(aliases)
    else:
      self._shared[original].extend(aliases)
    return self

  def factory(self, patterns: str | T.Iterable[str], factory: TensorLike | ParameterFactory):
    """Registers factory function to names matching any patterns."""
    if not callable(factory):
      factory = tensorlike_parameter_factory(factory)
    if isinstance(patterns, str):
      self._factories[patterns] = factory
    else:
      for pattern in patterns:
        self._factories[pattern] = factory
    return self

  def original(self, alias: str) -> str | None:
    """Returns the original name corresponding to `alias`, the same name if it was original, or None if not registered."""
    # Check if the provided alias is actually the original.
    if alias in self._shared: return alias
    # Search for the last-registered original with a compatible alias.
    for original, aliases in reversed(self._shared.items()):
      for pattern in aliases:
        if uts.in_glob_range(pattern, alias):
          return original
    return None

  def aliases(self, original: str) -> tuple[str, ...]:
    """Returns the aliases corresponding the `original`, or empty tuple if none registered."""
    return tuple(self._shared[original])

  def __getitem__(self, item: str | tuple[str, ParameterFactory | None]) -> torch.nn.Parameter:
    if isinstance(item, tuple):
      name, factory = item
    else:
      name, factory = item, None
    original = self.original(name)
    if original is not None:
      # Name is an alias for `original`.
      name = original
    if name in self:
      # Already constructed parameter.
      return super().__getitem__(name)
    if factory is None:
      # No factory supplied. Search for the last-registered factory.
      for pattern, fac in reversed(self._factories.items()):
        if uts.in_glob_range(pattern, name):
          factory = fac
          break
    if factory is None:
      # No factory supplied or registered.
      raise LookupError(f'No factory supplied or registered for parameter: {name}')
    param = factory()
    self[name] = param
    return param

  def get(self, name: str, factory: ParameterFactory | None = None) -> torch.nn.Parameter:
    return self[name, factory]

  def set(self, name: str, param: torch.nn.Parameter) -> torch.nn.Parameter:
    self[name] = param
    return param

  def extra_repr(self) -> str:
    return '\n'.join([f'  ({k}): {_repr_param(v)}' for k, v in self.items()])
