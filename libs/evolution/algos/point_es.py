import typing as T

import numpy as np
import ray
from evolution import core, noises, optimizers, shaping


class WorkerState(T.TypedDict):
  noise: np.ndarray


class ScoreKwargs(T.TypedDict):
  param: np.ndarray
  mutation_scale: float | np.ndarray
  mutation_seed: int
  objective_seed: int


ScoreResult = tuple[int, float, float]  # mutation_seed, score_pos, score_neg


class TestKwargs(T.TypedDict):
  param: np.ndarray
  objective_seed: int


class PointEvolutionStrategyWorker(core.AlgorithmWorker[ScoreResult]):
  def initialize(self, objective: core.Objective, noise: np.ndarray):
    self.objective = objective
    self.noise = noises.NoiseTable(noise)

  def score(
    self,
    param: np.ndarray,
    mutation_scale: float | np.ndarray,
    mutation_seed: int,
    objective_seed: int,
  ) -> ScoreResult:
    mutation = self.noise.sample(mutation_seed, param.size)
    score_pos = self.objective.get_score(param + mutation_scale * mutation, objective_seed)
    score_neg = self.objective.get_score(param - mutation_scale * mutation, objective_seed)
    return mutation_seed, score_pos, score_neg

  def test(self, param: np.ndarray, objective_seed: int):
    info = {}
    score = self.objective.get_score(param, objective_seed, info)
    return score, info


class PointEvolutionStrategy(core.Algorithm[WorkerState, ScoreKwargs, ScoreResult, TestKwargs]):
  worker_cls = PointEvolutionStrategyWorker

  def __init__(
    self,
    population_size: int = 8,
    population_top_best: int | None = None,
    mutation_noise: T.Callable[[int], np.ndarray] = noises.gaussian,
    mutation_scale: float | np.ndarray = 1.,
    score_shaping: T.Callable[[np.ndarray], np.ndarray] = shaping.identity,
    gradient_scale: float | np.ndarray = 1.,
    parameter_decay: float = 0.,
    optimizer: optimizers.Optimizer | None = None,
    objective_sync: bool = True,
  ):
    self.population_size = population_size
    self.population_top_best = population_top_best or population_size
    self.mutation_noise = mutation_noise
    self.mutation_scale = mutation_scale
    self.score_shaping = score_shaping
    self.gradient_scale = gradient_scale
    self.parameter_decay = parameter_decay
    self.optimizer = optimizer or optimizers.SGD()
    self.objective_sync = objective_sync

  def initialize(self, objective, seed) -> tuple[list[np.ndarray], WorkerState]:
    self.objective = objective
    initial_params = objective.get_initial_params(seed, 1)
    assert len(initial_params) == 1
    self.param = initial_params[0]
    self.optimizer.initialize(self.param)
    noise_ref = ray.put(self.mutation_noise(seed))
    self.noise = noises.NoiseTable(ray.get(noise_ref))
    return [self.param], {'noise': noise_ref}

  def spawn(self, epoch, seed) -> T.Iterable[ScoreKwargs]:
    param = ray.put(self.param)
    mutation_scale = self.mutation_scale
    if isinstance(mutation_scale, np.ndarray):
      mutation_scale = ray.put(mutation_scale)
    objective_sync = self.objective_sync
    for child in range(self.population_size):
      mutation_seed = noises.multiseed((3, 5, 5), (seed, epoch, child))
      objective_seed = noises.multiseed((3, 5, 5), (seed, epoch, 0 if objective_sync else child))
      yield {
        'param': param,
        'mutation_scale': mutation_scale,
        'mutation_seed': mutation_seed,
        'objective_seed': objective_seed,
      }

  def select(self, epoch, seed, results) -> core.AlgorithmTrainData:
    results = list(results)
    results.sort(key=lambda x: max(x[1], x[2]), reverse=True)  # Largest pos/neg scores first.
    results = results[:self.population_top_best]
    assert len(results) > 0
    seeds, scores_pos, scores_neg = zip(*results)
    scores_raw = np.vstack((scores_pos, scores_neg))
    scores_shaped = self.score_shaping(scores_raw)  # Shape (2, len(results)).
    scores_diff = scores_shaped[0, :] - scores_shaped[1, :]  # Score diff = pos - neg.

    param = self.param
    grad = np.zeros(1)
    for i in range(len(results)):
      mutation = self.noise.sample(seeds[i], param.size)
      grad = grad + scores_diff[i] * mutation
    grad = grad * self.gradient_scale / len(results)
    update = self.optimizer.step(grad) - self.parameter_decay * param
    self.param = param + update

    return core.AlgorithmTrainData(
      params=[self.param],
      stats={
        'scores/raw': scores_raw, 'scores/shaped': scores_shaped
      },
      artifacts={
        'scores/raw': scores_raw,
        'scores/shaped': scores_shaped,
        'scores/diff': scores_diff,
        'grad': grad,
        'update': update,
      },
    )

  def test(self, epoch, seed, size) -> T.Iterable[TestKwargs]:
    param = ray.put(self.param)
    for trial in range(size):
      objective_seed = noises.multiseed((3, 5, 3), (seed, epoch, trial))
      yield {'param': param, 'objective_seed': objective_seed}
