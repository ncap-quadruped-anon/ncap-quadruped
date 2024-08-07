import abc
import dataclasses
import os
import pathlib
import time
import typing as T

import numpy as np
import numpy.typing as npt
import ray
import tqdm
import utils.logger as utlog


class Objective(abc.ABC):
  def initialize(self):
    pass

  @abc.abstractmethod
  def get_initial_params(self, seed: int, size: int) -> list[np.ndarray]:
    raise NotImplementedError

  @abc.abstractmethod
  def get_score(self, param: np.ndarray, seed: int, info: dict[str, T.Any] = {}) -> float:
    """Return an evaluation of this objective with the given `param`."""
    raise NotImplementedError

  @abc.abstractmethod
  def get_state(self) -> T.TypedDict:
    """Return the state of this objective."""
    raise NotImplementedError

  @abc.abstractmethod
  def set_state(self, state: T.TypedDict) -> None:
    """Update the state of this objective."""
    raise NotImplementedError

  def on_run_start(self, *, algorithm: 'Algorithm', workers: list):
    """Callback executed by the `Trainer` at the start of the run."""
    pass

  def on_run_end(self, *, algorithm: 'Algorithm', workers: list):
    """Callback executed by the `Trainer` before the end of the run."""
    pass

  def on_train_epoch_start(self, *, epoch: int, algorithm: 'Algorithm', workers: list):
    """Callback executed by the `Trainer` before the start of a train epoch."""
    pass

  def on_train_epoch_end(
    self, *, epoch: int, algorithm: 'Algorithm', workers: list, train_data: 'AlgorithmTrainData'
  ):
    """Callback executed by the `Trainer` after the end of a train epoch."""
    pass

  def on_test_epoch_start(self, *, epoch: int, algorithm: 'Algorithm', workers: list):
    """Callback executed by the `Trainer` before the start of a test epoch."""
    pass

  def on_test_epoch_end(
    self, *, epoch: int, algorithm: 'Algorithm', workers: list, test_data: 'AlgorithmTestData'
  ):
    """Callback executed by the `Trainer` after the end of a test epoch."""
    pass

  def on_save(
    self,
    *,
    epoch: int,
    algorithm: 'Algorithm',
    workers: list,
    train_data: 'AlgorithmTrainData',
    test_data: 'AlgorithmTestData | None',
    log_dir: pathlib.Path
  ):
    """Callback executed by the `Trainer` after the end of a save epoch."""
    return {}


@dataclasses.dataclass
class AlgorithmTrainData:
  params: list[np.ndarray]
  stats: dict[str, float | int | str | list[float | int] | np.ndarray]
  artifacts: dict[str, list[float | int] | np.ndarray]


@dataclasses.dataclass
class AlgorithmTestData:
  scores: list[float]
  infos: list[dict[str, T.Any]]
  stats: dict[str, float | int | str | list[float | int] | np.ndarray]
  artifacts: dict[str, list[float | int] | np.ndarray]


WorkerState = T.TypeVar('WorkerState', bound=T.TypedDict)
ScoreKwargs = T.TypeVar('ScoreKwargs', bound=T.TypedDict)
ScoreResult = T.TypeVar('ScoreResult')
TestKwargs = T.TypeVar('TestKwargs', bound=T.TypedDict)
TestResult = tuple[float, dict[str, T.Any]]


class AlgorithmWorker(abc.ABC, T.Generic[ScoreResult]):
  """An algorithm's distributed logic for scoring parameter variations. Each worker process stores
  a separate instance of this class."""
  @abc.abstractmethod
  def initialize(self, objective: Objective, **worker_state) -> None:
    self.objective = objective

  @abc.abstractmethod
  def score(self, **score_kwargs) -> ScoreResult:
    raise NotImplementedError

  @abc.abstractmethod
  def test(self, **test_kwargs) -> TestResult:
    raise NotImplementedError


class Algorithm(abc.ABC, T.Generic[WorkerState, ScoreKwargs, ScoreResult, TestKwargs]):
  """An algorithm's centralized logic for spawning parameter variations and selecting parameter
  updates from the scored variations. The trainer process stores a single instance of this class."""

  # The `AlgorithmWorker` that performs the distributed logic for this `Algorithm`.
  worker_cls = AlgorithmWorker

  @abc.abstractmethod
  def initialize(self, objective: Objective, seed: int) -> tuple[list[np.ndarray], WorkerState]:
    raise NotImplementedError

  @abc.abstractmethod
  def spawn(self, epoch: int, seed: int) -> T.Iterable[ScoreKwargs]:
    raise NotImplementedError

  @abc.abstractmethod
  def select(self, epoch: int, seed: int, results: list[ScoreResult]) -> AlgorithmTrainData:
    raise NotImplementedError

  @abc.abstractmethod
  def test(self, epoch: int, seed: int, size: int) -> T.Iterable[TestKwargs]:
    raise NotImplementedError


@ray.remote(num_cpus=1)
class TrainerWorker:
  def __init__(
    self,
    objective_cls: T.Type[Objective],
    objective_kwargs: dict[str, T.Any],
    algorithm_worker_cls: T.Type[AlgorithmWorker],
    logger_kwargs: dict[str, T.Any],
  ):
    self.objective = objective_cls(**objective_kwargs)
    self.algorithm = algorithm_worker_cls()
    self.logger = utlog.Logger(**logger_kwargs)

  def initialize(self, **worker_state):
    self.objective.initialize()
    self.algorithm.initialize(self.objective, **worker_state)

  def train(self, **score_kwargs) -> T.Any:
    """Execute a train step by delegating to the algorithm's `score` function."""
    try:
      return self.algorithm.score(**score_kwargs)
    except:
      self.logger.exception(f'Error on worker train step with inputs: {score_kwargs}')
      raise

  def test(self, **test_kwargs) -> TestResult:
    """Execute a test step by delegating to the algorithm's `test` function."""
    try:
      return self.algorithm.test(**test_kwargs)
    except:
      self.logger.exception(f'Error on worker test step with inputs: {test_kwargs}')
      raise

  def get_objective_state(self) -> T.Any:
    """Return the state of this worker's objective, which can be useful for collecting information
    from the copies of the objective on different workers."""
    return self.objective.get_state()

  def set_objective_state(self, state: T.Any) -> None:
    """Update the state of this worker's objective, which can be useful for altering/synchonizing information from the copies of the objective on different workers."""
    self.objective.set_state(state)


class Trainer:
  """A `Trainer` encapsulates functionality for the train/test loop and data logging. It maximizes
  the given `Objective` using the given `Algorithm`. For computational efficiency, it performs
  scoring of parameter variants in distributed worker processes (using the Ray library)."""
  def __init__(
    self,
    algorithm: Algorithm,
    objective_cls: T.Type[Objective],
    objective_kwargs: dict[str, T.Any] = {},
    epochs: int = 1000,
    seed: int = 0,
    ray_options: dict[str, T.Any] = {},
    num_workers: int = 1,
    worker_options: dict[str, T.Any] = {},
    save_freq: int = 100,
    test_freq: int = 1,
    test_size: int = 5,
    show_progress: bool = False,
    logger: os.PathLike | str | utlog.Logger = './outputs',
  ):
    self.algorithm = algorithm
    self.objective = objective_cls(**objective_kwargs)
    self.epochs = epochs
    self.seed = seed
    self.test_freq = test_freq
    self.test_size = test_size
    self.save_freq = save_freq
    self.show_progress = show_progress
    self.logger = logger if isinstance(logger, utlog.Logger) else utlog.Logger(logger)

    # Initialize the workers.
    ray_options.setdefault('ignore_reinit_error', True)
    ray.init(**ray_options)
    logger_kwargs = dict(
      log_dir=self.logger.log_dir,
      name=self.logger.name,
      level=self.logger.level,
    )
    self.workers = [
      (
        TrainerWorker.options(**worker_options)  # type: ignore
        .remote(objective_cls, objective_kwargs, algorithm.worker_cls, logger_kwargs)
      ) for _ in range(num_workers)
    ]
    self.pool = ray.util.ActorPool(self.workers)

  def __del__(self):
    ray.shutdown()

  def initialize(self):
    self.objective.initialize()
    self.initial_params, worker_state = self.algorithm.initialize(self.objective, self.seed)
    ray.get([worker.initialize.remote(**worker_state) for worker in self.workers])

  def run(self) -> list[np.ndarray]:
    start_time = time.time()
    self.logger.info(f'Started run for {self.epochs} epochs')
    self.objective.on_run_start(algorithm=self.algorithm, workers=self.workers)

    params = self.initial_params
    self.logger.info(f'Initialized {len(params)} params with shapes {[p.shape for p in params]}')

    train_data = AlgorithmTrainData(params=params, stats={}, artifacts={})
    test_data = self._test(0, start_time)
    self._save(0, train_data, test_data)

    for epoch in tqdm.trange(1, self.epochs + 1, desc='epoch', disable=not self.show_progress):
      try:
        train_data = self._train(epoch, start_time)
        test_data = self._test(epoch, start_time)
        self._save(epoch, train_data, test_data)
        params = train_data.params
      except:
        self.logger.exception(f'Epoch {epoch}: Error occurred')
        raise

    self.objective.on_run_end(algorithm=self.algorithm, workers=self.workers)
    run_time = time.time() - start_time
    self.logger.info(f'Completed run for {self.epochs} epochs in {run_time:.2f} seconds')

    return params

  def _train(self, epoch: int, start_time: float):
    # Run train step (every epoch).
    train_time = time.time()
    self.objective.on_train_epoch_start(epoch=epoch, algorithm=self.algorithm, workers=self.workers)
    results = self.pool.map_unordered(
      lambda worker,
      score_kwargs: worker.train.remote(**score_kwargs),  # type: ignore
      list(self.algorithm.spawn(epoch, self.seed))
    )
    results = [x for x in results if x is not None]
    data = self.algorithm.select(epoch, self.seed, results)
    self.objective.on_train_epoch_end(
      epoch=epoch, algorithm=self.algorithm, workers=self.workers, train_data=data
    )

    # Save stats.
    curr_time = time.time()
    self.logger.table('train.csv').store({
      'train/epochs': epoch,
      'train/seconds': curr_time - start_time,
      'train/epoch_seconds': curr_time - train_time,
    }).store(
      data.stats, prefix='train/'
    ).save()
    self.logger.info(f'Epoch {epoch}: Completed train step')

    return data

  def _test(self, epoch: int, start_time: float):
    # Run train step (before first epoch, after every `test_freq` epochs, after last epoch).
    if (self.test_freq > 0 and epoch % self.test_freq == 0) or epoch == 0 or epoch == self.epochs:
      test_time = time.time()
      self.objective.on_test_epoch_start(
        epoch=epoch, algorithm=self.algorithm, workers=self.workers
      )
      results = self.pool.map_unordered(
        lambda worker,
        test_kwargs: worker.test.remote(**test_kwargs),  # type: ignore
        list(self.algorithm.test(epoch, self.seed, self.test_size))
      )
      results = [x for x in results if x is not None]
      scores, infos = zip(*results)
      data = AlgorithmTestData(scores=scores, infos=infos, stats={'scores': scores}, artifacts={})
      self.objective.on_test_epoch_end(
        epoch=epoch, algorithm=self.algorithm, workers=self.workers, test_data=data
      )

      # Save stats.
      curr_time = time.time()
      self.logger.table('test.csv').store({
        'test/epochs': epoch,
        'test/seconds': curr_time - start_time,
        'test/epoch_seconds': curr_time - test_time,
      }).store(
        data.stats, prefix='test/'
      ).save()
      self.logger.info(f'Epoch {epoch}: Completed test step')

      return data
    return None

  def _save(self, epoch: int, train_data: AlgorithmTrainData, test_data: AlgorithmTestData | None):
    # Save parameters/artifacts (after every `save_freq` epochs, after last epoch).
    if (self.save_freq > 0 and epoch % self.save_freq == 0) or epoch == 0 or epoch == self.epochs:
      self.objective.on_save(
        epoch=epoch,
        algorithm=self.algorithm,
        workers=self.workers,
        train_data=train_data,
        test_data=test_data,
        log_dir=self.logger.log_dir,
      )
      params = train_data.params
      self.logger.file('params').save(f'{epoch}.pkl', params)
      for k, v in train_data.artifacts.items():
        self.logger.file('train', k).save(f'{epoch}.pkl', v)
      if test_data is not None:
        for k, v in test_data.artifacts.items():
          self.logger.file('test', k).save(f'{epoch}.pkl', v)
      self.logger.info(f'Epoch {epoch}: Saved parameters and artifacts')
