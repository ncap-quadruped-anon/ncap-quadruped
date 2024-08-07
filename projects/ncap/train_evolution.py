import dataclasses
import os
import pathlib
import signal
import typing as T

import evolution
import hydra
import omegaconf as oc
import ray
import utils.logger as utlog

import projects.ncap.src.evolution as ncap_evo
import tools.infra  # pylint: disable=unused-import


@dataclasses.dataclass
class TrainEvolutionConfig:
  algorithm: evolution.Algorithm
  objective: ncap_evo.TonicTorchRollout
  trainer: evolution.Trainer  # Without keys: algorithm, objective_cls, objective_kwargs
  output_dir: str


@hydra.main(version_base='1.3', config_path='configs')
def main(cfg: oc.DictConfig) -> None:
  # Override SIGTERM handler so spawned processes don't adopt Submitit's bypass behavior.
  # See: https://github.com/facebookincubator/submitit/issues/1677
  signal.signal(signal.SIGTERM, lambda *args, **kwargs: exit(0))

  # Setup compute and output directory.
  ray.init(num_cpus=len(os.sched_getaffinity(0)), ignore_reinit_error=True)
  output_dir = pathlib.Path(cfg.output_dir)
  logger = utlog.Logger(output_dir)
  logger.info(f'Initialized ray with resources: {ray.available_resources()}')
  logger.info(f'Initialized output directory: {output_dir}')

  # Start trainer.
  algorithm: evolution.Algorithm = hydra.utils.instantiate(cfg.algorithm)
  objective_cls: T.Type[ncap_evo.TonicTorchRollout] = hydra.utils.get_class(cfg.objective._target_)
  objective_kwargs: dict[str, T.Any] = {
    k: v
    for k, v in cfg.objective.items()
    if k not in ('_target_', 'env_builder', 'model_builder')
  }
  env_builder_cfg = cfg.objective.env_builder
  env_builder = lambda: hydra.utils.instantiate(env_builder_cfg)()
  model_builder_cfg = cfg.objective.model_builder
  model_builder = lambda: hydra.utils.instantiate(model_builder_cfg)()
  trainer: evolution.Trainer = hydra.utils.instantiate(
    cfg.trainer,
    algorithm,
    objective_cls,
    objective_kwargs={
      'env_builder': env_builder,
      'model_builder': model_builder,
      **objective_kwargs,
    },
    logger=logger,
  )
  trainer.initialize()
  trainer.run()


if __name__ == '__main__':
  main()
