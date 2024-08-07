"""This module facilitates efficient sampling of noise by precomputive values in a noise table."""
import typing as T

import numpy as np


def multiseed(lengths: T.Sequence[int], seeds: T.Sequence[int]) -> int:
  """Combine multiple integer `seeds` into a single composite integer seed.

  Each input seed is allocated its corresponding `length` (in decimal digits) in the composite seed,
  and larger seeds are shortened via the modulo operator. The composite seed is prefixed by 1 to
  ensure consistent zero-padding. For example:
  ```
  multiseed(lengths=(2, 3, 4), seeds=(2, 333, 10004))
  >>> 1023330000  # 1-02-333-0004
  ```
  """
  assert len(lengths) == len(seeds) != 0
  seed = 1
  for i in range(len(lengths)):
    seed = seed * 10**lengths[i] + seeds[i] % 10**lengths[i]
  return seed


def gaussian(seed: int, size: int = 25_000_000, dtype: T.Type[np.floating] = np.float32):
  return np.random.default_rng(seed).standard_normal(size, dtype=dtype)


def bernoulli(seed: int, size: int = 25_000_000, dtype: T.Type[np.floating] = np.float32):
  return np.random.default_rng(seed).binomial(1, 0.5, size).astype(dtype) * 2 - 1


def uniform(seed: int, size: int = 25_000_000, dtype: T.Type[np.floating] = np.float32):
  return np.random.default_rng(seed).uniform(-1, 1, size).astype(dtype)


class NoiseTable:
  """A noise table speeds up sampling of noise by precomputing a large block and storing it in
  shared memory."""
  def __init__(self, noise: np.ndarray):
    self.noise = noise

  def sample(self, seed: int, size: int):
    """Sample a block of noise with `size` elements determined by the given `seed`. The same noise
    block will be return for the same value of `seed`."""
    idx = np.random.default_rng(seed).integers(0, len(self.noise) - size + 1)
    return self.noise[idx:idx + size]
