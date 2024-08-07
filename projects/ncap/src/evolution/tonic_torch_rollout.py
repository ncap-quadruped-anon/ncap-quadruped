import dataclasses
import logging
import os
import pathlib
import typing as T

import evolution
import gym.core
import numpy as np
import ray
import torch
import torch.nn.utils
import utils.io
from tonic.torch import models, normalizers

from ..quadruped.models import unflat

logger = logging.getLogger(__name__)


def as_tensor(x: np.ndarray | dict[str, np.ndarray], dtype=torch.float32):
  if isinstance(x, dict):
    return {k: torch.as_tensor(v, dtype=dtype) for k, v in x.items()}
  return torch.as_tensor(x, dtype=dtype)


def as_numpy(x: torch.Tensor | dict[str, torch.Tensor]):
  if isinstance(x, dict):
    return {k: v.numpy() for k, v in x.items()}
  return x.numpy()


Model = models.ActorOnly | unflat.UnflatActorOnly
Normalizer = normalizers.MeanStd | unflat.UnflatMeanStd | unflat.UnflatNormalizer


@dataclasses.dataclass
class TonicTorchRolloutCheckpoint:
  epoch: int
  env: gym.core.Env
  model: Model


def save_checkpoint(
  log_dir: pathlib.Path,
  epoch: int,
  model: Model,
  env_builder: T.Callable[[], gym.core.Env] | None = None,
  model_builder: T.Callable[[], Model] | None = None,
):
  if epoch == 0 and env_builder:
    utils.io.write_file(log_dir / 'checkpoint/env_builder.pkl', env_builder)
  if epoch == 0 and model_builder:
    utils.io.write_file(log_dir / 'checkpoint/model_builder.pkl', model_builder)
  torch.save(model.state_dict(), log_dir / f'checkpoint/{epoch}.pt')


def load_checkpoint(
  log_dir: pathlib.Path,
  epoch: T.Literal['first', 'last'] | int = 'last',
  seed: int = 0,
) -> TonicTorchRolloutCheckpoint:
  if isinstance(epoch, str):
    epochs = sorted([int(path.stem) for path in log_dir.glob('checkpoint/*.pt')])
    if len(epochs) == 0:
      raise FileNotFoundError(f'No checkpoints found in: {log_dir}')
    elif epoch == 'first':
      epoch = epochs[0]
    elif epoch == 'last':
      epoch = epochs[-1]
    else:
      raise ValueError(f'Invalid checkpoint epoch: {epoch}')

  env_builder = utils.io.read_file(log_dir / 'checkpoint/env_builder.pkl')
  model_builder = utils.io.read_file(log_dir / 'checkpoint/model_builder.pkl')

  # Ensure reproducibility if model/env uses randomness in initialization.
  np.random.seed(seed)
  torch.manual_seed(seed)
  env: gym.core.Env = env_builder()
  model: Model = model_builder()
  model.initialize(env.observation_space, env.action_space)
  model.load_state_dict(torch.load(log_dir / f'checkpoint/{epoch}.pt'))
  return TonicTorchRolloutCheckpoint(epoch=epoch, env=env, model=model)


class StateFromWorker(T.TypedDict):
  normalizer_new_stats: T.Any | None


class StateToWorker(T.TypedDict):
  normalizer_state: T.Any | None


class TonicTorchRollout(evolution.Objective):
  def __init__(
    self,
    env_builder: T.Callable[[], gym.core.Env],
    model_builder: T.Callable[[], Model],
    rollouts: int = 1,
    timeouts: int = 10_000,
  ):
    self.env_builder = env_builder
    self.model_builder = model_builder
    self.rollouts = rollouts
    self.timeouts = timeouts

    self.env = env_builder()
    self.model = model_builder()

  def initialize(self):
    self.model.initialize(self.env.observation_space, self.env.action_space)
    self.model_param_trainable = [p for p in self.model.parameters() if p.requires_grad == True]

  def get_state(self) -> StateFromWorker:
    observation_normalizer: Normalizer | None = self.model.observation_normalizer
    return {
      'normalizer_new_stats': observation_normalizer and observation_normalizer.get_new_stats(),
    }

  def set_state(self, state: StateToWorker) -> None:
    observation_normalizer: Normalizer | None = self.model.observation_normalizer
    if observation_normalizer:
      observation_normalizer.set_state(state['normalizer_state'])

  def on_run_start(self, *, algorithm: evolution.Algorithm, workers: list):
    # Log information about the environment and model.
    logger.info(f'env:\n{self.env}')
    logger.info(f'env.observation_space:\n{self.env.observation_space}')
    logger.info(f'env.action_space:\n{self.env.action_space}')
    logger.info(f'model:\n{self.model}')

  def on_train_epoch_end(
    self,
    *,
    epoch: int,
    algorithm: evolution.Algorithm,
    workers: list,
    train_data: evolution.Algorithm,
  ):
    # Gather, combine, and broadcast model normalizer statistics.
    observation_normalizer: Normalizer | None = self.model.observation_normalizer
    if not observation_normalizer: return
    states_from_workers = ray.get([worker.get_objective_state.remote() for worker in workers])
    for state in states_from_workers:
      observation_normalizer.record_new_stats(state['normalizer_new_stats'])
    observation_normalizer.update()
    state_to_workers = ray.put({'normalizer_state': observation_normalizer.get_state()})
    ray.get([worker.set_objective_state.remote(state_to_workers) for worker in workers])

  def on_save(
    self,
    *,
    epoch: int,
    algorithm: evolution.Algorithm,
    workers: list,
    train_data: evolution.AlgorithmTrainData,
    test_data: evolution.AlgorithmTestData | None,
    log_dir: pathlib.Path,
  ):
    param = train_data.params[0]
    torch.nn.utils.vector_to_parameters(  # type: ignore
      torch.as_tensor(param, dtype=torch.float32), self.model_param_trainable,
    )
    save_checkpoint(
      log_dir=log_dir,
      epoch=epoch,
      model=self.model,
      env_builder=self.env_builder,
      model_builder=self.model_builder,
    )

  def get_initial_params(self, seed: int, size: int) -> list[np.ndarray]:
    # Convert seed to standard `int` seed between [0, 2**32) due to env/torch constraints.
    rng = np.random.default_rng(seed)
    seed32 = rng.integers(2**32).item()
    torch.manual_seed(seed32)

    initial_params = []
    with torch.no_grad():
      for _ in range(size):
        model = self.model_builder()
        model.initialize(self.env.observation_space, self.env.action_space)
        param = torch.nn.utils.parameters_to_vector(self.model_param_trainable)  # type: ignore
        initial_params.append(param.numpy())
    return initial_params

  def get_updated_params(self, params: list[np.ndarray]) -> list[np.ndarray]:
    if getattr(self.model.actor, 'constrainer', None) is None: return params
    updated_params = []
    with torch.no_grad():
      for param in params:
        torch.nn.utils.vector_to_parameters(  # type: ignore
          torch.as_tensor(param, dtype=torch.float32), self.model_param_trainable,
        )
        self.model.actor.constrainer.constrain()
        param = torch.nn.utils.parameters_to_vector(self.model_param_trainable)  # type: ignore
        updated_params.append(param.numpy())
    return updated_params

  def get_score(self, param: np.ndarray, seed: int, info: dict[str, T.Any] = {}) -> float:
    env = self.env
    model = self.model

    # The info dict stores information for debugging and analysis. If a dict is supplied as an
    # argument, that object will be directly mutated.
    info['rollouts'] = []

    average_return = 0.
    with torch.no_grad():
      torch.nn.utils.vector_to_parameters(  # type: ignore
        torch.as_tensor(param, dtype=torch.float32), self.model_param_trainable,
      )
      if getattr(model.actor, 'constrainer', None) is not None:
        model.actor.constrainer.constrain()

      # Convert seed to standard `int` seed between [0, 2**32) due to env/torch constraints.
      rng = np.random.default_rng(seed)
      seed32 = rng.integers(2**32).item()
      torch.manual_seed(seed32)
      env.seed(seed32)

      for rollout in range(1, self.rollouts + 1):
        obs, done = env.reset(), False
        observations = [obs]
        actions = []
        rewards = []
        dones = []
        steps = 0

        model.reset()  # Enables both stateful and stateless models.
        while not done and steps < self.timeouts:
          action = as_numpy(model(as_tensor(obs)))
          obs, reward, done, _ = env.step(action)
          observations.append(obs)
          actions.append(action)
          dones.append(done)
          steps += 1
          if reward is None: break  # In Tonic: reward=None on timeouts, done=True on terminations.
          rewards.append(reward)
          average_return += reward

        if model.observation_normalizer:
          model.observation_normalizer.record(observations)

        info['rollouts'].append({
          'observations': observations,
          'actions': actions,
          'rewards': rewards,
          'dones': dones,
        })

    average_return /= self.rollouts
    info['average_return'] = average_return

    return average_return


def init_linear_zeros_(module):
  if isinstance(module, torch.nn.Linear):
    torch.nn.init.zeros_(module.weight)
    torch.nn.init.zeros_(module.bias)
