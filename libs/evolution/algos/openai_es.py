import typing as T

import numpy as np

from .. import optimizers, shaping
from . import point_es


class OpenAIEvolutionStrategy(point_es.PointEvolutionStrategy):
  def __init__(
    self,
    mutation_scale: float | np.ndarray = 0.02,
    score_shaping: T.Callable[[np.ndarray], np.ndarray] = shaping.scaled_rank,
    gradient_scale: float | np.ndarray = 1.,
    parameter_decay: float = 0.,
    optimizer: optimizers.Optimizer | None = None,
    **kwargs,
  ):
    super().__init__(
      mutation_scale=mutation_scale,
      score_shaping=score_shaping,
      gradient_scale=gradient_scale / mutation_scale,
      parameter_decay=parameter_decay,
      optimizer=optimizer or optimizers.Adam(),
      **kwargs,
    )
