import typing as T

import numpy as np

from .. import optimizers, shaping
from . import point_es


class AugmentedRandomSearchEvolutionStrategy(point_es.PointEvolutionStrategy):
  def __init__(
    self,
    mutation_scale: float | np.ndarray = 0.02,
    score_shaping: T.Callable[[np.ndarray], np.ndarray] = shaping.z_scores,
    gradient_scale: float | np.ndarray = 1.,
    optimizer: optimizers.Optimizer | None = None,
    **kwargs,
  ):
    super().__init__(
      mutation_scale=mutation_scale,
      gradient_scale=gradient_scale * mutation_scale,
      score_shaping=score_shaping,
      optimizer=optimizer or optimizers.SGD(learning_rate=1),
      **kwargs,
    )
